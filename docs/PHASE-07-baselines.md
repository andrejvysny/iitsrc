# PHASE 7: Cloud Baselines

**Duration**: ~1-1.5 hours
**Schedule**: Day 2, 17:00-18:00 (parallel with late Phase 5/6)
**Dependencies**: Phase 1 (API keys), Phase 2 (data), Phase 3 (annotations), Phase 4 (baselines.py)
**Blocks**: Phase 8 (analysis needs baseline comparison data)

---

## 1. Objective

Run all 5 cloud LLM baselines (GPT-4o, Claude Sonnet, Qwen-72B, Llama-70B, Mistral-Large) on the 100 test pages. These serve as the upper bound for extraction quality that local models aim to match.

---

## 2. Tasks

### 2.1 Run Cloud Baselines for Both Ideas

For Idea B: run each cloud model on 4 pruning strategies (best format).
For Idea C: run each cloud model on best preprocessing only (1 config per model).

**Shared runner**: `shared/baselines.py` (implemented in Phase 4).

```python
# run_cloud_baselines.py

import json
import csv
import time
from pathlib import Path
from tqdm import tqdm

from shared.schemas import get_schema
from shared.metrics import compute_all_metrics
from shared.baselines import CLOUD_MODELS, run_cloud_model
from shared.utils import determine_domain
from shared.preprocessing import html_to_clean_text

# Import pruning from Idea B
import sys
sys.path.insert(0, str(Path("idea-b-schema-pruning/src")))
from pruning import apply_pruning, convert_format

DATA_DIR = Path("idea-b-schema-pruning/data")
RESULTS_DIR = Path("shared/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pages():
    pages = []
    for html_file in sorted((DATA_DIR / "raw_html").glob("*.html")):
        ann_file = DATA_DIR / "annotations" / f"{html_file.stem}.json"
        if ann_file.exists():
            pages.append({
                "name": html_file.stem,
                "html": html_file.read_text(encoding="utf-8", errors="replace"),
                "annotation": json.loads(ann_file.read_text()),
                "domain": determine_domain(html_file.name),
            })
    return pages


def run_for_idea_b(pages):
    """Run cloud baselines across pruning strategies for Idea B."""
    results = []
    pruning_strategies = ["none", "generic", "heuristic", "semantic"]
    best_format = "flat_json"

    for model_name, model_id in CLOUD_MODELS.items():
        for strategy in pruning_strategies:
            print(f"  {model_name} / {strategy}...")
            for page in tqdm(pages, desc=f"{model_name}/{strategy}"):
                try:
                    pruned = apply_pruning(page["html"], strategy, domain=page["domain"])
                    content = convert_format(pruned, best_format)

                    schema = get_schema(page["domain"])
                    result = run_cloud_model(content, schema, model_id)

                    gold = page["annotation"]["data"]
                    metrics = compute_all_metrics(
                        result["parsed"], gold, schema, page["html"], result["raw_output"]
                    )

                    results.append({
                        "page_id": page["name"],
                        "domain": page["domain"],
                        "pruning": strategy,
                        "format": best_format,
                        "model": model_name,
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "json_valid": metrics["json_valid"],
                        "schema_valid": metrics["schema_valid"],
                        "hallucination_rate": metrics["hallucination_rate"],
                        "tokens_in": result.get("input_tokens"),
                        "latency_s": result.get("latency_s"),
                    })

                    time.sleep(0.5)  # Rate limit protection
                except Exception as e:
                    print(f"    ERROR: {page['name']}: {e}")

    return results


def run_for_idea_c(pages):
    """Run cloud baselines with best preprocessing for Idea C comparison."""
    results = []
    best_pruning = "semantic"
    best_format = "flat_json"

    for model_name, model_id in CLOUD_MODELS.items():
        print(f"  {model_name}...")
        for page in tqdm(pages, desc=model_name):
            try:
                pruned = apply_pruning(page["html"], best_pruning, domain=page["domain"])
                content = convert_format(pruned, best_format)

                schema = get_schema(page["domain"])
                result = run_cloud_model(content, schema, model_id)

                gold = page["annotation"]["data"]
                metrics = compute_all_metrics(
                    result["parsed"], gold, schema, page["html"], result["raw_output"]
                )

                results.append({
                    "page_id": page["name"],
                    "domain": page["domain"],
                    "model": model_name,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "json_valid": metrics["json_valid"],
                    "schema_valid": metrics["schema_valid"],
                    "hallucination_rate": metrics["hallucination_rate"],
                    "tokens_in": result.get("input_tokens"),
                    "latency_s": result.get("latency_s"),
                })

                time.sleep(0.5)
            except Exception as e:
                print(f"    ERROR: {page['name']}: {e}")

    return results


def main():
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")

    # Idea B baselines (5 models × 4 pruning × 100 pages = 2000 calls)
    print("\n=== Idea B Cloud Baselines ===")
    idea_b_results = run_for_idea_b(pages)
    with open(RESULTS_DIR / "cloud_baselines_idea_b.csv", "w", newline="") as f:
        if idea_b_results:
            writer = csv.DictWriter(f, fieldnames=idea_b_results[0].keys())
            writer.writeheader()
            writer.writerows(idea_b_results)
    print(f"Idea B: {len(idea_b_results)} results saved")

    # Idea C baselines (5 models × 100 pages = 500 calls)
    print("\n=== Idea C Cloud Baselines ===")
    idea_c_results = run_for_idea_c(pages)
    with open(RESULTS_DIR / "cloud_baselines_idea_c.csv", "w", newline="") as f:
        if idea_c_results:
            writer = csv.DictWriter(f, fieldnames=idea_c_results[0].keys())
            writer.writeheader()
            writer.writerows(idea_c_results)
    print(f"Idea C: {len(idea_c_results)} results saved")


if __name__ == "__main__":
    main()
```

### 2.2 Budget Tracking

```python
# Inline budget tracker (add to baselines.py or separate)

PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-3-5-haiku-20241022": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    # OpenRouter prices vary, approximate
    "qwen-72b": {"input": 0.39 / 1_000_000, "output": 0.39 / 1_000_000},
    "llama-70b": {"input": 0.52 / 1_000_000, "output": 0.75 / 1_000_000},
    "mistral-large": {"input": 2.00 / 1_000_000, "output": 6.00 / 1_000_000},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost for a single call."""
    pricing = PRICING.get(model, {"input": 0.001, "output": 0.003})
    return input_tokens * pricing["input"] + output_tokens * pricing["output"]
```

---

## 3. Cost Estimates

### 3.1 Idea B Cloud Baselines (2000 API calls)

| Model | Calls | ~Input tok/call | ~Output tok/call | Est. Cost |
|-------|-------|-----------------|------------------|-----------|
| GPT-4o | 400 | 800 | 200 | ~$1.60 |
| Claude Sonnet | 400 | 800 | 200 | ~$2.16 |
| Qwen-72B (OR) | 400 | 800 | 200 | ~$0.19 |
| Llama-70B (OR) | 400 | 800 | 200 | ~$0.23 |
| Mistral-Large (OR) | 400 | 800 | 200 | ~$1.12 |
| **Total** | **2000** | | | **~$5.30** |

### 3.2 Idea C Cloud Baselines (500 API calls)

| Model | Calls | Est. Cost |
|-------|-------|-----------|
| GPT-4o | 100 | ~$0.40 |
| Claude Sonnet | 100 | ~$0.54 |
| Qwen-72B | 100 | ~$0.05 |
| Llama-70B | 100 | ~$0.06 |
| Mistral-Large | 100 | ~$0.28 |
| **Total** | **500** | **~$1.33** |

### 3.3 Budget Summary

| Category | Est. Cost |
|----------|-----------|
| Annotation (Phase 3) | ~$2 |
| Cloud baselines (Phase 7) | ~$6.60 |
| Buffer for retries/debug | ~$3 |
| **Total estimated** | **~$12** |

Well within $10-50 budget.

---

## 4. Execution Strategy

### 4.1 Priority Order

Run cheapest models first (to validate pipeline), expensive models last:

1. Qwen-72B via OpenRouter (~$0.05/100 pages) — fast, cheap
2. Llama-70B via OpenRouter (~$0.06/100 pages) — fast, cheap
3. Mistral-Large via OpenRouter (~$0.28/100 pages) — moderate
4. GPT-4o (~$0.40/100 pages) — primary baseline
5. Claude Sonnet (~$0.54/100 pages) — secondary baseline

### 4.2 Rate Limiting

- OpenAI: 500 RPM (gpt-4o) → no issue for 100 pages
- Anthropic: 50 RPM (Sonnet) → add 1.2s sleep between calls
- OpenRouter: varies by model, ~60 RPM → add 1s sleep

### 4.3 Parallelization

Can run different models in parallel (different API endpoints, no interference).
Run via separate terminal tabs or Python multiprocessing.

---

## 5. Acceptance Criteria

- [ ] All 5 cloud models × 100 pages = 500 results for Idea C
- [ ] All 5 cloud models × 4 pruning × 100 pages = 2000 results for Idea B
- [ ] JSON validity rate > 90% for all cloud models
- [ ] GPT-4o macro F1 > 0.80 (expected strong baseline)
- [ ] Total API cost < $15
- [ ] Results saved as CSV with all metric columns
- [ ] Cost tracking logged per model

---

## 6. Verification

```python
import pandas as pd

# Idea B
df_b = pd.read_csv("shared/results/cloud_baselines_idea_b.csv")
print(f"Idea B: {len(df_b)} rows, {df_b['model'].nunique()} models")
for model in df_b["model"].unique():
    m = df_b[df_b["model"] == model]
    print(f"  {model}: F1={m['f1'].mean():.3f}, JSON%={m['json_valid'].mean()*100:.0f}%")

# Idea C
df_c = pd.read_csv("shared/results/cloud_baselines_idea_c.csv")
print(f"\nIdea C: {len(df_c)} rows")
for model in df_c["model"].unique():
    m = df_c[df_c["model"] == model]
    print(f"  {model}: F1={m['f1'].mean():.3f}")
```

---

## 7. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits hit | Medium | Low | Add sleep between calls. Retry with backoff |
| OpenRouter model unavailable | Low | Medium | Use alternative models (Qwen-72B → Qwen-32B, etc.) |
| Budget exceeded | Low | Medium | Track costs per call. Stop if >$20 |
| Slow API responses | Medium | Low | Set timeout=60s. Skip stuck calls |
| Claude response format issues | Low | Low | Add robust JSON parsing (already in utils.py) |

---

## 8. Output Files

```
shared/results/
├── cloud_baselines_idea_b.csv    # 5 models × 4 pruning × 100 pages
├── cloud_baselines_idea_c.csv    # 5 models × 100 pages
└── cost_log.json                 # Per-call cost tracking
```

---

## 9. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Run OpenRouter models (3 models × 100 pages) | 15 min |
| Run GPT-4o (100 pages) | 10 min |
| Run Claude Sonnet (100 pages + rate limiting) | 15 min |
| Run Idea B variant (4 pruning × above) | 45 min total |
| Verify results + fix errors | 10 min |
| **Total** | **~1-1.5 hours** |

**Note**: Can overlap with Phase 5/6 local experiments since cloud API calls are independent.
