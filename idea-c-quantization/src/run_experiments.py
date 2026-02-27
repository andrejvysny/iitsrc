"""Idea C experiment runner: model × quantization level."""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root and local src to path
_src_dir = str(Path(__file__).resolve().parent)
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _src_dir)
sys.path.insert(0, _root_dir)

from shared.dataset import load_and_cache_sample
from shared.metrics import compute_all_metrics
from shared.utils import count_tokens

RESULTS_DIR = Path(os.environ.get(
    "IITSRC_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results"),
))
RESULTS_CSV = RESULTS_DIR / "experiments.csv"

CSV_COLUMNS = [
    "page_id", "model", "quant",
    "schema_keys", "schema_complexity",
    "f1", "precision", "recall", "exact_match",
    "json_valid", "schema_valid", "hallucination_rate",
    "tokens_in", "latency_s", "model_size_gb",
]


def truncate_to_tokens(content: str, max_tokens: int) -> str:
    """Truncate content to fit within token budget."""
    tokens = count_tokens(content)
    if tokens <= max_tokens:
        return content
    ratio = max_tokens / tokens
    cut_len = int(len(content) * ratio * 0.95)
    return content[:cut_len]


def load_completed(csv_path: Path) -> set[tuple[str, str, str]]:
    """Load completed experiment keys from CSV."""
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["page_id"], row["model"], row["quant"])
                completed.add(key)
    return completed


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single result row to CSV."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def run_single(
    page: dict,
    model: str,
    quant: str,
    extract_fn,
) -> dict[str, Any] | None:
    """Run a single model×quant experiment. Returns result dict or None."""
    schema = page["schema"]
    gold = page["gold"]
    content = page["content_markdown"]

    # Truncate based on schema size
    n_props = sum(1 for _ in schema.get("properties", {}))
    max_tokens = 7000 if n_props >= 10 else 3500
    content = truncate_to_tokens(content, max_tokens)

    try:
        result = extract_fn(model, quant, content, schema)
    except Exception as e:
        print(f"  Extraction error ({model}/{quant}): {e}")
        return None

    predicted = result.get("parsed") or {}
    raw = result.get("raw_output", "")

    metrics = compute_all_metrics(predicted, gold, schema, content, raw)

    return {
        "page_id": page["id"],
        "model": model,
        "quant": quant,
        "schema_keys": page.get("schema_keys", 0),
        "schema_complexity": page.get("schema_complexity_score", 0.0),
        "f1": round(metrics["f1"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "exact_match": int(metrics["exact_match"]),
        "json_valid": int(metrics["json_valid"]),
        "schema_valid": int(metrics["schema_valid"]),
        "hallucination_rate": round(metrics["hallucination_rate"], 4),
        "tokens_in": result.get("tokens_in", 0),
        "latency_s": round(result.get("latency_s", 0), 2),
        "model_size_gb": result.get("model_size_gb", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea C: Quantization impact experiments")
    parser.add_argument("--limit", type=int, default=500, help="Max pages to process")
    parser.add_argument("--models", nargs="+", default=None, help="Models to test (default: all)")
    parser.add_argument("--quants", nargs="+", default=None, help="Quant levels (default: all)")
    args = parser.parse_args()

    from inference import load_and_extract, get_all_variants, MODEL_QUANTS, unload_model

    # Load data (uses Markdown directly — no HTML needed)
    print("Loading sample records...")
    records = load_and_cache_sample(args.limit)
    print(f"Loaded {len(records)} records")

    # Get model×quant variants
    all_variants = get_all_variants()
    if args.models:
        all_variants = [(m, q) for m, q in all_variants if m in args.models]
    if args.quants:
        all_variants = [(m, q) for m, q in all_variants if q in args.quants]
    print(f"Model variants: {len(all_variants)}")

    # Setup results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed(RESULTS_CSV)
    print(f"Already completed: {len(completed)} configurations")

    # Build experiment list
    configs = []
    for model, quant in all_variants:
        for page in records:
            key = (page["id"], model, quant)
            if key not in completed:
                configs.append((page, model, quant))

    print(f"Remaining experiments: {len(configs)}")

    # Sort by (model, quant) for sequential loading
    configs.sort(key=lambda x: (x[1], x[2]))

    current_variant = None
    for page, model, quant in tqdm(configs, desc="Experiments"):
        variant = (model, quant)
        if variant != current_variant:
            current_variant = variant
            tqdm.write(f"\nLoading: {model} / {quant}")

        result = run_single(page, model, quant, load_and_extract)
        if result:
            append_result(RESULTS_CSV, result)

    unload_model()
    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
