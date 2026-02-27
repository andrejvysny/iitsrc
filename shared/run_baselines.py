"""Cloud baseline runner: 5 models × N pages on Markdown content."""

import argparse
import csv
import time
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from shared.baselines import CLOUD_MODELS, run_cloud_model
from shared.dataset import load_and_cache_sample
from shared.metrics import compute_all_metrics
from shared.utils import count_tokens

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "cloud_baselines.csv"

CSV_COLUMNS = [
    "page_id", "model", "context_type",
    "schema_keys", "schema_complexity",
    "f1", "precision", "recall", "exact_match",
    "json_valid", "schema_valid", "hallucination_rate",
    "tokens_in", "latency_s", "cost_usd",
]

# Approx pricing per 1M input tokens (USD)
INPUT_PRICING = {
    "gpt-4o": 2.50,
    "claude-sonnet": 3.00,
    "qwen-72b": 0.27,
    "llama-70b": 0.52,
    "mistral-large": 2.00,
}

# Rate limit delays (seconds between calls)
RATE_DELAYS = {
    "gpt-4o": 0.5,
    "claude-sonnet": 1.2,
    "qwen-72b": 1.0,
    "llama-70b": 1.0,
    "mistral-large": 1.0,
}


def load_completed(csv_path: Path) -> set[tuple[str, str]]:
    """Load completed (page_id, model) pairs from CSV."""
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["page_id"], row["model"]))
    return completed


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    """Append single result row to CSV."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def estimate_cost(tokens_in: int, model: str) -> float:
    """Estimate cost in USD for input tokens."""
    rate = INPUT_PRICING.get(model, 1.0)
    return tokens_in * rate / 1_000_000


def truncate_to_tokens(content: str, max_tokens: int) -> str:
    """Truncate content to fit within token budget."""
    tokens = count_tokens(content)
    if tokens <= max_tokens:
        return content
    ratio = max_tokens / tokens
    return content[: int(len(content) * ratio * 0.95)]


def run_single(page: dict, model: str) -> dict[str, Any] | None:
    """Run single cloud model extraction. Returns result dict or None."""
    schema = page["schema"]
    gold = page["gold"]
    content = page["content_markdown"]

    # Truncate: cloud models handle more context, but cap at 12k tokens
    n_props = sum(1 for _ in schema.get("properties", {}))
    max_tokens = 12000 if n_props >= 10 else 8000
    content = truncate_to_tokens(content, max_tokens)

    try:
        result = run_cloud_model(content, schema, model)
    except Exception as e:
        print(f"  API error ({model}): {e}")
        return None

    predicted = result.get("parsed") or {}
    raw = result.get("raw_output", "")
    tokens_in = result.get("input_tokens", 0)

    metrics = compute_all_metrics(predicted, gold, schema, content, raw)

    return {
        "page_id": page["id"],
        "model": model,
        "context_type": "markdown",
        "schema_keys": page.get("schema_keys", 0),
        "schema_complexity": page.get("schema_complexity_score", 0.0),
        "f1": round(metrics["f1"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "exact_match": int(metrics["exact_match"]),
        "json_valid": int(metrics["json_valid"]),
        "schema_valid": int(metrics["schema_valid"]),
        "hallucination_rate": round(metrics["hallucination_rate"], 4),
        "tokens_in": tokens_in,
        "latency_s": round(result.get("latency_s", 0), 2),
        "cost_usd": round(estimate_cost(tokens_in, model), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud baseline experiments")
    parser.add_argument("--limit", type=int, default=500, help="Max pages")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to test (default: all). Options: {list(CLOUD_MODELS.keys())}",
    )
    args = parser.parse_args()

    models = args.models or list(CLOUD_MODELS.keys())
    invalid = [m for m in models if m not in CLOUD_MODELS]
    if invalid:
        print(f"Unknown models: {invalid}. Valid: {list(CLOUD_MODELS.keys())}")
        sys.exit(1)

    print("Loading sample records...")
    records = load_and_cache_sample(args.limit)
    print(f"Loaded {len(records)} records")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed(RESULTS_CSV)
    print(f"Already completed: {len(completed)} configurations")

    # Build experiment list: model × pages, sorted by model for rate limiting
    configs = []
    for model in models:
        for page in records:
            if (page["id"], model) not in completed:
                configs.append((page, model))
    print(f"Remaining experiments: {len(configs)}")

    if not configs:
        print("Nothing to do.")
        return

    total_cost = 0.0
    errors = 0

    for page, model in tqdm(configs, desc="Baselines"):
        result = run_single(page, model)
        if result:
            append_result(RESULTS_CSV, result)
            total_cost += result["cost_usd"]
        else:
            errors += 1

        # Rate limiting
        delay = RATE_DELAYS.get(model, 1.0)
        time.sleep(delay)

    print(f"\nDone. Results: {RESULTS_CSV}")
    print(f"Total estimated cost: ${total_cost:.4f}")
    print(f"Errors: {errors}/{len(configs)}")


if __name__ == "__main__":
    main()
