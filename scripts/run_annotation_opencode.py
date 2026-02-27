"""Annotate HTML files using opencode CLI with dual-model consensus."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from shared.metrics import hallucination_rate, schema_valid, values_match
from shared.preprocessing import clean_html
from shared.schemas import get_schema, get_schema_field_names
from shared.utils import parse_json_safe, timer

DATA_DIR = Path(__file__).resolve().parents[1] / "idea-b-schema-pruning" / "data"
RAW_DIR = DATA_DIR / "raw_html"
ANN_DIR = DATA_DIR / "annotations"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"

PRIMARY_MODEL = "openai/gpt-5.1-codex-mini"
SECONDARY_MODEL = "google/gemini-2.5-flash"
MAX_CONTENT_TOKENS = 20_000
SUBPROCESS_TIMEOUT = 120


def load_manifest() -> dict[str, dict]:
    """Load manifest.jsonl into {filename: entry} dict."""
    entries: dict[str, dict] = {}
    if not MANIFEST_PATH.exists():
        return entries
    for line in MANIFEST_PATH.read_text().splitlines():
        if line.strip():
            try:
                entry = json.loads(line)
                entries[entry["filename"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def detect_domain(filename: str) -> str:
    """Detect domain from filename prefix."""
    if filename.startswith("ecom_"):
        return "ecommerce"
    elif filename.startswith("realestate_"):
        return "realestate"
    raise ValueError(f"Cannot detect domain from filename: {filename}")


def build_prompt(schema: dict) -> str:
    """Build extraction prompt string for opencode CLI argument."""
    schema_json = json.dumps(schema, indent=2)
    return (
        "Extract structured product data from the attached HTML content.\n"
        "Return ONLY valid JSON matching this schema:\n\n"
        f"{schema_json}\n\n"
        "Rules:\n"
        "- Use null for fields not found in the content\n"
        "- price must be a number (no currency symbols)\n"
        "- availability must be one of: in_stock, out_of_stock, unknown\n"
        "- specs: key product attributes only (Author, ISBN, Condition, etc.)\n"
        "- Do NOT include shipping, returns, or seller info in specs\n"
        "- Do NOT invent values not present in the source"
    )


def run_opencode(
    model: str, temp_path: Path, prompt: str
) -> tuple[str, str, int]:
    """Run opencode CLI and return (stdout, stderr, returncode)."""
    cmd = [
        "opencode", "run",
        "--model", model,
        "--file", str(temp_path),
        prompt,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    return result.stdout, result.stderr, result.returncode


def call_model(
    model: str, temp_path: Path, prompt: str, page_name: str
) -> dict:
    """Call a single model via opencode. Returns per-model result dict."""
    parsed = None
    raw_output = ""
    exec_time = 0.0
    error = None

    for attempt in range(2):
        try:
            with timer() as t:
                stdout, stderr, rc = run_opencode(model, temp_path, prompt)
            exec_time += t["elapsed"]
        except subprocess.TimeoutExpired:
            logging.warning("%s [%s]: timeout (attempt %d)", page_name, model, attempt + 1)
            error = f"timeout after {SUBPROCESS_TIMEOUT}s"
            continue

        if rc != 0:
            error = stderr.strip()[:200]
            logging.warning("%s [%s]: exit %d (attempt %d): %s", page_name, model, rc, attempt + 1, error)
            continue

        raw_output = stdout
        if not raw_output.strip():
            error = "empty stdout"
            logging.warning("%s [%s]: empty stdout (attempt %d)", page_name, model, attempt + 1)
            continue

        error = None
        parsed = parse_json_safe(raw_output)
        if parsed is not None:
            break

        if attempt == 0:
            logging.warning("%s [%s]: JSON parse failed, retrying...", page_name, model)
            error = "json_parse_failed"

    return {
        "response": parsed,
        "raw_output": raw_output,
        "execution_time": round(exec_time, 2),
        "error": error,
    }


def compute_consensus(
    r1: dict, r2: dict, fields: list[str]
) -> tuple[dict, dict, list[str]]:
    """Field-level consensus from two model results.

    Returns (consensus_data, agreement_scores, disagreement_fields).
    If only one model produced valid JSON, uses that model's output.
    """
    resp1, resp2 = r1["response"], r2["response"]

    if resp1 is None and resp2 is None:
        return {f: None for f in fields}, {f: 0.0 for f in fields}, list(fields)

    if resp1 is None:
        return resp2, {f: 1.0 for f in fields}, []
    if resp2 is None:
        return resp1, {f: 1.0 for f in fields}, []

    consensus: dict = {}
    scores: dict = {}
    disagreements: list[str] = []

    for f in fields:
        v1 = resp1.get(f)
        v2 = resp2.get(f)
        if values_match(v1, v2):
            consensus[f] = v1 if v1 is not None else v2
            scores[f] = 1.0
        else:
            # Prefer primary model on disagreement
            consensus[f] = v1
            scores[f] = 0.0
            disagreements.append(f)

    return consensus, scores, disagreements


def annotate_file(
    html_path: Path,
    primary_model: str,
    secondary_model: str | None,
    manifest: dict[str, dict],
    delay: float,
    dry_run: bool = False,
) -> dict | None:
    """Annotate a single HTML file. Returns annotation dict or None on failure."""
    page_name = html_path.stem

    try:
        domain = detect_domain(html_path.name)
    except ValueError as e:
        logging.warning("Skipping %s: %s", html_path.name, e)
        return None

    schema = get_schema(domain)
    fields = get_schema_field_names(domain)
    prompt = build_prompt(schema)

    manifest_entry = manifest.get(html_path.name, {})
    source_url = manifest_entry.get("url", "")

    html_content = html_path.read_text(errors="ignore")
    content, json_ld = clean_html(html_content, max_tokens=MAX_CONTENT_TOKENS)

    if dry_run:
        mode = "dual" if secondary_model else "single"
        logging.info(
            "[DRY RUN] %s: domain=%s content_len=%d mode=%s",
            page_name, domain, len(content), mode,
        )
        return None

    # Write cleaned content to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=f"opencode_input_{page_name}_",
        delete=False,
    )
    try:
        tmp.write(content)
        tmp.close()
        temp_path = Path(tmp.name)

        with timer() as total_t:
            # Primary model
            r1 = call_model(primary_model, temp_path, prompt, page_name)

            if secondary_model:
                time.sleep(delay)
                r2 = call_model(secondary_model, temp_path, prompt, page_name)
                consensus, scores, disagreements = compute_consensus(r1, r2, fields)
            else:
                # Single-model mode
                r2 = None
                consensus = r1["response"] or {}
                scores = {f: 1.0 for f in fields}
                disagreements = []

        # Validate consensus
        is_valid = bool(consensus) and schema_valid(consensus, schema)
        h_rate = hallucination_rate(consensus, content) if consensus else 0.0
        total_time = total_t["elapsed"]

        # per_model dict (compatible with export_dataset.py)
        per_model = {primary_model: r1}
        if r2 is not None:
            per_model[secondary_model] = r2

        any_parsed = r1["response"] is not None or (r2 is not None and r2["response"] is not None)
        agree_pct = sum(1 for s in scores.values() if s == 1.0) / len(scores) if scores else 0.0

        annotation = {
            "data": consensus,
            "domain": domain,
            "page_name": page_name,
            "source_url": source_url,
            "agreement_scores": scores,
            "per_model": per_model,
            "escalated_fields": disagreements,
            "validation": {
                "schema_valid": is_valid,
                "hallucination_rate": round(h_rate, 4),
                "parse_success": any_parsed,
            },
            "json_ld": json_ld,
            "source_text_length": len(content),
            "total_execution_time": round(total_time, 2),
        }

        status = "valid" if is_valid else ("parsed" if any_parsed else "FAILED")
        if secondary_model:
            logging.info(
                "%s: %s agree=%.0f%% disagree=%s time=%.1fs",
                page_name, status, agree_pct * 100,
                disagreements or "none", total_time,
            )
        else:
            logging.info(
                "%s: %s time=%.1fs h_rate=%.2f",
                page_name, status, total_time, h_rate,
            )
        return annotation

    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate HTML files using opencode CLI (dual-model consensus)"
    )
    parser.add_argument("--model", default=PRIMARY_MODEL, help="Primary model ID")
    parser.add_argument("--secondary-model", default=SECONDARY_MODEL, help="Secondary model ID for consensus")
    parser.add_argument("--single", action="store_true", help="Single-model mode (skip consensus)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between calls")
    parser.add_argument("--limit", type=int, default=0, help="Max files (0=all)")
    parser.add_argument("--file", type=str, default="", help="Single file by name")
    parser.add_argument("--force", action="store_true", help="Re-annotate existing")
    parser.add_argument("--dry-run", action="store_true", help="Print without executing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check opencode is available
    if not args.dry_run and not shutil.which("opencode"):
        logging.error("opencode CLI not found in PATH. Install: https://github.com/opencode-ai/opencode")
        sys.exit(1)

    secondary = None if args.single else args.secondary_model
    mode_str = f"{args.model} (single)" if args.single else f"{args.model} + {secondary}"

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    # Collect files
    if args.file:
        html_files = [RAW_DIR / args.file]
        if not html_files[0].exists():
            logging.error("File not found: %s", html_files[0])
            sys.exit(1)
    else:
        html_files = sorted(RAW_DIR.glob("*.html"))

    # Skip already annotated
    if not args.force:
        html_files = [
            f for f in html_files
            if not (ANN_DIR / f"{f.stem}.json").exists()
        ]

    if args.limit > 0:
        html_files = html_files[: args.limit]

    if not html_files:
        logging.info("No files to annotate.")
        return

    logging.info(
        "Annotating %d files with %s (delay=%.1fs)",
        len(html_files), mode_str, args.delay,
    )

    completed = 0
    failed = 0

    for html_path in tqdm(html_files, desc="Annotating"):
        annotation = annotate_file(
            html_path, args.model, secondary, manifest, args.delay, args.dry_run,
        )

        if args.dry_run:
            continue

        if annotation is None:
            failed += 1
            continue

        if not annotation["validation"]["parse_success"]:
            failed += 1

        out_path = ANN_DIR / f"{html_path.stem}.json"
        out_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False))
        completed += 1

        # Rate limit between files
        if html_path != html_files[-1]:
            time.sleep(args.delay)

    if not args.dry_run:
        logging.info(
            "Done: %d/%d completed, %d failed",
            completed, len(html_files), failed,
        )


if __name__ == "__main__":
    main()
