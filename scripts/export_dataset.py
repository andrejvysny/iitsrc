"""Export annotation JSONs to flat CSV + JSONL dataset."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.schemas import get_schema_description

DATA_DIR = Path(__file__).resolve().parents[1] / "idea-b-schema-pruning" / "data"
ANN_DIR = DATA_DIR / "annotations"
EXPORT_DIR = DATA_DIR / "exports"

# Model name mapping for short keys in per_model
MODEL_NAMES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "qwen-2.5-72b-instruct": "openrouter/qwen/qwen-2.5-72b-instruct",
}


def load_annotations() -> list[dict]:
    """Load all annotation JSONs."""
    annotations = []
    for f in sorted(ANN_DIR.glob("*.json")):
        if f.name == "annotation_stats.json":
            continue
        try:
            annotations.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}")
    return annotations


def flatten_rows(annotations: list[dict]) -> list[dict]:
    """Create one row per model per page + one consensus row."""
    rows = []

    for ann in annotations:
        domain = ann["domain"]
        page_name = ann["page_name"]
        schema_str = get_schema_description(domain)

        # Per-model rows
        for model_key, model_data in ann.get("per_model", {}).items():
            response = model_data.get("response")
            rows.append({
                "page_name": page_name,
                "domain": domain,
                "source_url": ann.get("source_url", ""),
                "llm_model": model_key,
                "source": "model",
                "schema": schema_str,
                "response": json.dumps(response) if response else "",
                "response_is_valid": bool(response is not None),
                "execution_time": model_data.get("execution_time", 0),
                "tokens": model_data.get("tokens", 0),
                "cost": model_data.get("cost", 0),
                "error": model_data.get("error", ""),
            })

        # Consensus row
        rows.append({
            "page_name": page_name,
            "domain": domain,
            "source_url": ann.get("source_url", ""),
            "llm_model": "consensus",
            "source": "consensus",
            "schema": schema_str,
            "response": json.dumps(ann.get("data", {})),
            "response_is_valid": ann.get("validation", {}).get("schema_valid", False),
            "execution_time": ann.get("total_execution_time", 0),
            "tokens": 0,
            "cost": ann.get("total_cost_estimate", 0),
            "error": "",
        })

    return rows


def export_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations()
    if not annotations:
        print("No annotations found in", ANN_DIR)
        return

    rows = flatten_rows(annotations)

    csv_path = EXPORT_DIR / "dataset.csv"
    jsonl_path = EXPORT_DIR / "dataset.jsonl"

    export_csv(rows, csv_path)
    export_jsonl(rows, jsonl_path)

    # Summary
    n_pages = len(annotations)
    n_models = len(set(r["llm_model"] for r in rows if r["source"] == "model"))
    print(f"Exported {len(rows)} rows ({n_pages} pages x {n_models} models + consensus)")
    print(f"  CSV:   {csv_path}")
    print(f"  JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
