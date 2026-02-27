"""Run annotation pipeline on crawled HTML files."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from shared.annotator import Annotator

DATA_DIR = Path(__file__).resolve().parents[1] / "idea-b-schema-pruning" / "data"
RAW_DIR = DATA_DIR / "raw_html"
ANN_DIR = DATA_DIR / "annotations"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"


def load_manifest() -> dict[str, dict]:
    """Load manifest.jsonl into {filename: entry} dict."""
    entries = {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run annotation pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Max pages to annotate (0=all)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between pages (seconds)")
    parser.add_argument("--file", type=str, default="", help="Annotate single file by name")
    parser.add_argument("--force", action="store_true", help="Re-annotate even if JSON exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ANN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    # Collect files to process
    if args.file:
        html_files = [RAW_DIR / args.file]
        if not html_files[0].exists():
            logging.error("File not found: %s", html_files[0])
            sys.exit(1)
    else:
        html_files = sorted(RAW_DIR.glob("*.html"))

    # Filter already annotated
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

    logging.info("Annotating %d files", len(html_files))

    # Track stats
    stats = {
        "total": len(html_files),
        "completed": 0,
        "failed": 0,
        "total_cost": 0.0,
        "total_time": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }

    # Group by domain for annotator reuse
    annotators: dict[str, Annotator] = {}

    for html_path in tqdm(html_files, desc="Annotating"):
        page_name = html_path.stem

        try:
            domain = detect_domain(html_path.name)
        except ValueError as e:
            logging.warning("Skipping %s: %s", html_path.name, e)
            stats["failed"] += 1
            continue

        if domain not in annotators:
            annotators[domain] = Annotator(domain, delay=1.0)
        annotator = annotators[domain]

        # Get source URL from manifest
        manifest_entry = manifest.get(html_path.name, {})
        source_url = manifest_entry.get("url", "")

        html_content = html_path.read_text(errors="ignore")

        try:
            result = annotator.annotate_page(html_content, page_name, source_url)
            out_path = ANN_DIR / f"{page_name}.json"
            out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

            stats["completed"] += 1
            stats["total_cost"] += result.total_cost_estimate
            stats["total_time"] += result.total_execution_time
            stats["files"].append({
                "page_name": page_name,
                "cost": round(result.total_cost_estimate, 6),
                "time": round(result.total_execution_time, 2),
                "schema_valid": result.validation.get("schema_valid", False),
                "escalated": len(result.escalated_fields),
            })

            logging.info(
                "%s: cost=$%.4f time=%.1fs valid=%s escalated=%d",
                page_name,
                result.total_cost_estimate,
                result.total_execution_time,
                result.validation.get("schema_valid"),
                len(result.escalated_fields),
            )

        except Exception as e:
            logging.error("Failed %s: %s", page_name, e)
            stats["failed"] += 1

        # Rate limit between pages
        time.sleep(args.delay)

    # Save stats
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["total_cost"] = round(stats["total_cost"], 4)
    stats["total_time"] = round(stats["total_time"], 2)
    stats_path = ANN_DIR / "annotation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    logging.info(
        "Done: %d/%d completed, %d failed, $%.4f total cost",
        stats["completed"], stats["total"], stats["failed"], stats["total_cost"],
    )


if __name__ == "__main__":
    main()
