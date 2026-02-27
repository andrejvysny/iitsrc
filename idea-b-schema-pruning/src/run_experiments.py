"""Idea B experiment runner: pruning strategy × format × model."""

import argparse
import csv
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
from shared.html_fetcher import fetch_all_html, get_fetch_stats
from shared.metrics import compute_all_metrics
from shared.utils import count_tokens

from pruning import apply_pruning, convert_format
from extract import extract, unload_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "experiments.csv"

STRATEGIES = ["none", "generic", "heuristic", "semantic"]
FORMATS = ["simplified_html", "markdown", "flat_json"]
MODELS = ["qwen2.5-3b", "llama-3.2-3b", "phi-3.5-mini"]

CSV_COLUMNS = [
    "page_id", "pruning", "format", "model",
    "schema_keys", "schema_complexity",
    "f1", "precision", "recall", "exact_match",
    "json_valid", "schema_valid", "hallucination_rate",
    "tokens_in", "latency_s",
]



def truncate_to_tokens(content: str, max_tokens: int) -> str:
    """Truncate content to fit within token budget."""
    tokens = count_tokens(content)
    if tokens <= max_tokens:
        return content
    # Approximate: cut by ratio
    ratio = max_tokens / tokens
    cut_len = int(len(content) * ratio * 0.95)
    return content[:cut_len]


def load_completed(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """Load completed experiment keys from CSV."""
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["page_id"], row["pruning"], row["format"], row["model"])
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
    html: str,
    strategy: str,
    fmt: str,
    model_name: str,
    apply_pruning_fn,
    convert_format_fn,
    extract_fn,
) -> dict[str, Any] | None:
    """Run a single experiment configuration. Returns result dict or None."""
    schema = page["schema"]
    gold = page["gold"]

    try:
        # Prune
        pruned = apply_pruning_fn(html, strategy, schema=schema)
        # Convert format
        content = convert_format_fn(pruned, fmt)
    except Exception as e:
        import traceback
        print(f"  Pruning/format error ({strategy}/{fmt}): {e}")
        traceback.print_exc()
        return None

    # Truncate to fit context
    n_props = sum(1 for _ in schema.get("properties", {}))
    max_tokens = 7000 if n_props >= 10 else 3500
    content = truncate_to_tokens(content, max_tokens)

    try:
        result = extract_fn(content, schema, model_name)
    except Exception as e:
        import traceback
        print(f"  Extraction error ({model_name}): {e}")
        traceback.print_exc()
        return None

    predicted = result.get("parsed") or {}
    raw = result.get("raw_output", "")

    metrics = compute_all_metrics(predicted, gold, schema, content, raw)

    return {
        "page_id": page["id"],
        "pruning": strategy,
        "format": fmt,
        "model": model_name,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea B: Schema-aware pruning experiments")
    parser.add_argument("--limit", type=int, default=500, help="Max pages to process")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to test")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, help="Pruning strategies")
    parser.add_argument("--formats", nargs="+", default=FORMATS, help="Output formats")
    args = parser.parse_args()

    # Load data
    print("Loading sample records...")
    records = load_and_cache_sample(args.limit)
    print(f"Loaded {len(records)} records")

    # Fetch HTML
    print("Fetching HTML...")
    html_results = fetch_all_html(records)
    stats = get_fetch_stats(html_results)
    print(f"HTML fetch: {stats['success']}/{stats['total']} success ({stats['success_rate']:.1%})")

    # Filter to pages with HTML
    pages_with_html = [
        rec for rec in records
        if html_results.get(rec["id"]) is not None
    ]
    print(f"Pages with HTML: {len(pages_with_html)}")

    # Setup results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed(RESULTS_CSV)
    print(f"Already completed: {len(completed)} configurations")

    # Build experiment matrix
    configs = []
    for strategy in args.strategies:
        for fmt in args.formats:
            for model_name in args.models:
                for page in pages_with_html:
                    key = (page["id"], strategy, fmt, model_name)
                    if key not in completed:
                        configs.append((page, strategy, fmt, model_name))

    print(f"Remaining experiments: {len(configs)}")

    # Run experiments (group by model for cache efficiency)
    configs.sort(key=lambda x: x[3])  # sort by model name

    current_model = None
    for page, strategy, fmt, model_name in tqdm(configs, desc="Experiments"):
        if model_name != current_model:
            current_model = model_name
            tqdm.write(f"\nSwitching to model: {model_name}")

        html = html_results[page["id"]]
        result = run_single(
            page, html, strategy, fmt, model_name,
            apply_pruning, convert_format, extract,
        )
        if result:
            append_result(RESULTS_CSV, result)

    unload_model()
    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
