"""Quick test: run annotator on 3 HTML pages and validate output."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.annotator import Annotator
from shared.schemas import get_schema

DATA_DIR = Path(__file__).resolve().parent.parent / "idea-b-schema-pruning" / "data" / "raw_html"

TEST_PAGES = [
    "ecom_ebay_electronics_001.html",
    "ecom_amazon_books_001.html",
    "ecom_ebay_computers_001.html",
]


def run_test() -> None:
    annotator = Annotator(domain="ecommerce", delay=1.0)
    schema = get_schema("ecommerce")
    required_fields = schema["required"]
    all_fields = list(schema["properties"].keys())

    for page_name in TEST_PAGES:
        path = DATA_DIR / page_name
        if not path.exists():
            print(f"SKIP {page_name} — not found")
            continue

        print(f"\n{'='*60}")
        print(f"PAGE: {page_name}")
        print(f"HTML size: {path.stat().st_size:,} bytes")
        print(f"{'='*60}")

        html = path.read_text(encoding="utf-8", errors="replace")
        result = annotator.annotate_page(html, page_name)
        rd = result.to_dict()

        # Print extracted data
        print("\nExtracted data:")
        for f in all_fields:
            val = rd["data"].get(f)
            display = val
            if isinstance(val, str) and len(val) > 80:
                display = val[:80] + "..."
            elif isinstance(val, list):
                display = f"[{len(val)} items]"
            print(f"  {f}: {display}")

        # Validation
        v = rd["validation"]
        print(f"\nValidation:")
        print(f"  schema_valid: {v['schema_valid']}")
        print(f"  rules_passed: {v['rules_passed']}")
        print(f"  source_grounded: {v['source_grounded']} (hallucination_rate={v['hallucination_rate']})")
        if v["issues"]:
            print(f"  issues: {v['issues']}")

        # Agreement
        print(f"\nModel agreement:")
        for f, score in rd["agreement_scores"].items():
            tag = "OK" if score == 1.0 else ("ESCALATED" if score == 0.5 else "DISAGREE")
            print(f"  {f}: {score} [{tag}]")

        if rd["escalated_fields"]:
            print(f"\nEscalated fields: {rd['escalated_fields']}")

        # Per-model raw responses
        print(f"\nPer-model:")
        for model, info in rd["per_model"].items():
            err = f" ERROR: {info['error']}" if info["error"] else ""
            print(f"  {model}: {info['execution_time']}s, {info['tokens']} tokens, ${info['cost']}{err}")

        # Check required fields are non-null
        missing = [f for f in required_fields if rd["data"].get(f) is None]
        if missing:
            print(f"\n  WARNING: missing required fields: {missing}")

        print(f"\nTotal: {rd['total_execution_time']}s, ${rd['total_cost_estimate']}")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    run_test()
