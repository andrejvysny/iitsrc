# PHASE 6: Idea C — Quantization Experiments

**Duration**: ~3-4 hours
**Schedule**: Day 2, 13:00-18:00
**Dependencies**: Phase 1, Phase 2, Phase 3, Phase 4, Phase 5 (best preprocessing determined)
**Blocks**: Phase 8 (analysis)

---

## 1. Objective

Download all GGUF quantization variants (7 levels × 4 models = 25 variants after RAM limits), implement unified inference pipeline, run all experiments on 100 pages, measure quality degradation across quantization levels.

---

## 2. Tasks

### 2.1 Download All GGUF Variants

**File**: `idea-c-quantization/src/download_models.py`

```python
"""Download all GGUF quantization variants for experiment."""

from huggingface_hub import hf_hub_download
from pathlib import Path
import os

MODELS_DIR = Path("models")

# Model sources and their quantization file naming patterns
DOWNLOADS = {
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "variants": {
            "f16":    "qwen2.5-3b-instruct-fp16.gguf",
            "q8_0":   "qwen2.5-3b-instruct-q8_0.gguf",
            "q6_k":   "qwen2.5-3b-instruct-q6_k.gguf",
            "q5_k_m": "qwen2.5-3b-instruct-q5_k_m.gguf",
            "q4_k_m": "qwen2.5-3b-instruct-q4_k_m.gguf",  # already downloaded
            "q3_k_m": "qwen2.5-3b-instruct-q3_k_m.gguf",
            "q2_k":   "qwen2.5-3b-instruct-q2_k.gguf",
        },
    },
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "variants": {
            "f16":    "Llama-3.2-3B-Instruct-f16.gguf",
            "q8_0":   "Llama-3.2-3B-Instruct-Q8_0.gguf",
            "q6_k":   "Llama-3.2-3B-Instruct-Q6_K.gguf",
            "q5_k_m": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
            "q4_k_m": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "q3_k_m": "Llama-3.2-3B-Instruct-Q3_K_M.gguf",
            "q2_k":   "Llama-3.2-3B-Instruct-Q2_K.gguf",
        },
    },
    "phi-3.5-mini": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "variants": {
            "f16":    "Phi-3.5-mini-instruct-f16.gguf",
            "q8_0":   "Phi-3.5-mini-instruct-Q8_0.gguf",
            "q6_k":   "Phi-3.5-mini-instruct-Q6_K.gguf",
            "q5_k_m": "Phi-3.5-mini-instruct-Q5_K_M.gguf",
            "q4_k_m": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
            "q3_k_m": "Phi-3.5-mini-instruct-Q3_K_M.gguf",
            "q2_k":   "Phi-3.5-mini-instruct-Q2_K.gguf",
        },
    },
    "mistral-7b": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
        "variants": {
            # No F16 or Q8 — too large for 24GB RAM with KV cache
            # "f16" → ~14 GB model + ~4 GB KV = ~18 GB (might work but tight)
            # "q8_0" → ~7.5 GB + ~4 GB KV = ~11.5 GB (should work)
            "q8_0":   "mistral-7b-instruct-v0.3.Q8_0.gguf",
            "q6_k":   "mistral-7b-instruct-v0.3.Q6_K.gguf",
            "q5_k_m": "mistral-7b-instruct-v0.3.Q5_K_M.gguf",
            "q4_k_m": "mistral-7b-instruct-v0.3.Q4_K_M.gguf",
            "q3_k_m": "mistral-7b-instruct-v0.3.Q3_K_M.gguf",
            "q2_k":   "mistral-7b-instruct-v0.3.Q2_K.gguf",
        },
    },
}

# Note: Exact filenames may differ — verify on HuggingFace repos before downloading.
# The above are best-effort guesses based on common GGUF naming conventions.
# IMPORTANT: Check actual filenames on each HuggingFace repo page.


def download_all():
    """Download all GGUF variants."""
    total = sum(len(m["variants"]) for m in DOWNLOADS.values())
    downloaded = 0

    for model_name, config in DOWNLOADS.items():
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for quant, filename in config["variants"].items():
            target = model_dir / f"{quant}.gguf"

            if target.exists():
                size_gb = target.stat().st_size / (1024**3)
                print(f"  SKIP {model_name}/{quant} (exists, {size_gb:.1f} GB)")
                downloaded += 1
                continue

            print(f"  Downloading {model_name}/{quant} ({filename})...")
            try:
                path = hf_hub_download(
                    repo_id=config["repo"],
                    filename=filename,
                    local_dir=str(model_dir),
                )
                # Rename to standardized name
                downloaded_path = model_dir / filename
                if downloaded_path.exists():
                    downloaded_path.rename(target)
                downloaded += 1
                size_gb = target.stat().st_size / (1024**3)
                print(f"    Done: {size_gb:.1f} GB")
            except Exception as e:
                print(f"    FAILED: {e}")

    print(f"\nDownloaded {downloaded}/{total} variants")


if __name__ == "__main__":
    download_all()
```

#### Download Size Estimates

| Model | Quant | Estimated Size |
|-------|-------|---------------|
| Qwen/Llama/Phi (3B) | F16 | ~6.0 GB |
| Qwen/Llama/Phi (3B) | Q8_0 | ~3.2 GB |
| Qwen/Llama/Phi (3B) | Q6_K | ~2.5 GB |
| Qwen/Llama/Phi (3B) | Q5_K_M | ~2.2 GB |
| Qwen/Llama/Phi (3B) | Q4_K_M | ~2.0 GB |
| Qwen/Llama/Phi (3B) | Q3_K_M | ~1.5 GB |
| Qwen/Llama/Phi (3B) | Q2_K | ~1.2 GB |
| Mistral (7B) | Q8_0 | ~7.5 GB |
| Mistral (7B) | Q6_K | ~5.8 GB |
| Mistral (7B) | Q5_K_M | ~5.0 GB |
| Mistral (7B) | Q4_K_M | ~4.4 GB |
| Mistral (7B) | Q3_K_M | ~3.3 GB |
| Mistral (7B) | Q2_K | ~2.5 GB |

**Total: ~80 GB** for all variants
**Total for 3B models only: ~55 GB**
**Critical path**: Start downloading in background during Phase 5

### 2.2 Implement `idea-c-quantization/src/inference.py`

```python
"""Unified inference across quantization levels."""

from llama_cpp import Llama
from pathlib import Path
import json

from shared.prompts import build_messages
from shared.utils import parse_json_response, Timer, count_tokens, get_model_size_gb
from shared.schemas import get_schema


MODELS_DIR = Path("models")

# Chat format mapping
CHAT_FORMATS = {
    "qwen2.5-3b": "chatml",
    "llama-3.2-3b": "llama-3",
    "phi-3.5-mini": "phi-3",
    "mistral-7b": "mistral-instruct",
}


def get_model_path(model_name: str, quant: str) -> Path:
    """Get path to a specific model variant."""
    return MODELS_DIR / model_name / f"{quant}.gguf"


def load_and_extract(
    model_name: str,
    quant: str,
    content: str,
    domain: str,
    n_ctx: int = 4096,
    max_tokens: int = 1024,
) -> dict:
    """Load a specific quant variant and run extraction.

    Important: Each call loads the model fresh to ensure clean state.
    For batch processing, the caller should manage model lifecycle.
    """
    model_path = get_model_path(model_name, quant)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_size = get_model_size_gb(str(model_path))
    schema = get_schema(domain)
    tokens_in = count_tokens(content)

    # Load model
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=False,
        chat_format=CHAT_FORMATS.get(model_name),
    )

    messages = [
        {"role": "system", "content": (
            "You are a structured data extraction assistant. "
            "Output ONLY valid JSON. Use null for missing fields."
        )},
        {"role": "user", "content": (
            f"JSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Web Content:\n{content}\n\nOutput:"
        )},
    ]

    with Timer() as t:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )

    raw_output = response["choices"][0]["message"]["content"]
    parsed = parse_json_response(raw_output)

    # Clean up model to free memory for next variant
    del llm

    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "latency_s": t.elapsed,
        "tokens_in": tokens_in,
        "tokens_out": response.get("usage", {}).get("completion_tokens"),
        "model": model_name,
        "quant": quant,
        "model_size_gb": model_size,
    }
```

### 2.3 Implement `idea-c-quantization/src/run_experiments.py`

```python
"""Run all Idea C experiments: 4 models × 7 quant levels × 100 pages."""

import json
import csv
from pathlib import Path
from tqdm import tqdm
import gc

from shared.schemas import get_schema
from shared.metrics import compute_all_metrics
from shared.utils import determine_domain

from inference import load_and_extract, get_model_path

# Use best preprocessing from Idea B (determined after Phase 5)
# Default assumption: schema-semantic pruning + flat_json format
import sys
sys.path.insert(0, str(Path("idea-b-schema-pruning/src")))
from pruning import apply_pruning, convert_format

DATA_DIR = Path("idea-b-schema-pruning/data")  # shared data
RESULTS_DIR = Path("idea-c-quantization/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Best preprocessing from Idea B
BEST_PRUNING = "semantic"  # Update after Phase 5 results
BEST_FORMAT = "flat_json"  # Update after Phase 5 results

MODELS_AND_QUANTS = {
    "qwen2.5-3b":   ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
    "llama-3.2-3b":  ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
    "phi-3.5-mini":  ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
    "mistral-7b":    ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
}


def load_pages() -> list[dict]:
    """Load all HTML pages and annotations."""
    pages = []
    html_dir = DATA_DIR / "raw_html"
    ann_dir = DATA_DIR / "annotations"

    for html_file in sorted(html_dir.glob("*.html")):
        ann_file = ann_dir / f"{html_file.stem}.json"
        if not ann_file.exists():
            continue

        pages.append({
            "name": html_file.stem,
            "html": html_file.read_text(encoding="utf-8", errors="replace"),
            "annotation": json.loads(ann_file.read_text()),
            "domain": determine_domain(html_file.name),
        })
    return pages


def preprocess_pages(pages: list[dict]) -> dict:
    """Preprocess all pages once (pruning + format conversion). Cache results."""
    preprocessed = {}
    for page in tqdm(pages, desc="Preprocessing"):
        pruned = apply_pruning(page["html"], BEST_PRUNING, domain=page["domain"])
        content = convert_format(pruned, BEST_FORMAT)
        preprocessed[page["name"]] = content
    return preprocessed


def run_all():
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")

    preprocessed = preprocess_pages(pages)
    print(f"Preprocessed {len(preprocessed)} pages")

    results = []
    csv_path = RESULTS_DIR / "experiments.csv"

    with open(csv_path, "w", newline="") as f:
        writer = None

        for model_name, quants in MODELS_AND_QUANTS.items():
            for quant in quants:
                model_path = get_model_path(model_name, quant)
                if not model_path.exists():
                    print(f"  SKIP {model_name}/{quant} (not downloaded)")
                    continue

                print(f"\n=== {model_name} / {quant} ===")

                for page in tqdm(pages, desc=f"{model_name}/{quant}"):
                    content = preprocessed[page["name"]]
                    try:
                        result = load_and_extract(
                            model_name, quant, content, page["domain"]
                        )

                        gold = page["annotation"]["data"]
                        schema = get_schema(page["domain"])
                        metrics = compute_all_metrics(
                            result["parsed"], gold, schema,
                            page["html"], result["raw_output"]
                        )

                        row = {
                            "page_id": page["name"],
                            "domain": page["domain"],
                            "model": model_name,
                            "quant": quant,
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "json_valid": metrics["json_valid"],
                            "schema_valid": metrics["schema_valid"],
                            "hallucination_rate": metrics["hallucination_rate"],
                            "tokens_in": result["tokens_in"],
                            "latency_s": result["latency_s"],
                            "model_size_gb": result["model_size_gb"],
                        }
                        results.append(row)

                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=row.keys())
                            writer.writeheader()
                        writer.writerow(row)
                        f.flush()

                    except Exception as e:
                        print(f"    ERROR: {page['name']} - {e}")

                # Force garbage collection between quant levels
                gc.collect()

    print(f"\nDone! {len(results)} experiments → {csv_path}")


if __name__ == "__main__":
    run_all()
```

---

## 3. Experiment Matrix

### 3.1 Full Matrix: 25 Model Variants

| # | Model | Quant | Bits | Est. Size (GB) | Fits 24GB? |
|---|-------|-------|------|-----------------|------------|
| 1 | qwen2.5-3b | F16 | 16 | 6.0 | Yes |
| 2 | qwen2.5-3b | Q8_0 | 8 | 3.2 | Yes |
| 3 | qwen2.5-3b | Q6_K | 6 | 2.5 | Yes |
| 4 | qwen2.5-3b | Q5_K_M | 5 | 2.2 | Yes |
| 5 | qwen2.5-3b | Q4_K_M | 4 | 2.0 | Yes |
| 6 | qwen2.5-3b | Q3_K_M | 3 | 1.5 | Yes |
| 7 | qwen2.5-3b | Q2_K | 2 | 1.2 | Yes |
| 8-14 | llama-3.2-3b | same 7 | | same | Yes |
| 15-21 | phi-3.5-mini | same 7 | | +0.3 GB each | Yes |
| 22 | mistral-7b | Q8_0 | 8 | 7.5 | Yes (tight) |
| 23 | mistral-7b | Q6_K | 6 | 5.8 | Yes |
| 24 | mistral-7b | Q5_K_M | 5 | 5.0 | Yes |
| 25 | mistral-7b | Q4_K_M | 4 | 4.4 | Yes |
| — | mistral-7b | Q3_K_M | 3 | 3.3 | Yes |
| — | mistral-7b | Q2_K | 2 | 2.5 | Yes |

**Note**: Mistral-7B F16 (~14 GB) excluded; KV cache + OS overhead would exceed 24 GB.

### 3.2 Time Estimation

Per model variant on 100 pages:
- 3B model Q4_K_M: ~10s/page → 17 min
- 3B model F16: ~15s/page → 25 min (larger, slower)
- 7B model Q4_K_M: ~20s/page → 33 min

Total: 25 variants × 20 min avg = ~8 hours

**Time reduction**:
1. Run 50 pages instead of 100 for all except Q4_K_M baseline → ~4.5 hours
2. Run only 3B models (21 variants) → ~6 hours
3. Run pilot (20 pages) for all, full (100) for key variants → ~3.5 hours

**Recommended**: Option 3 — pilot on 20 pages (all 25 variants), full run on Q4_K_M + Q8_0 + Q2_K per model

---

## 4. Key Measurements

For each model × quant × page:

| Metric | Source | Unit |
|--------|--------|------|
| F1 (macro) | `field_f1()` | 0.0-1.0 |
| Precision | `field_f1()` | 0.0-1.0 |
| Recall | `field_f1()` | 0.0-1.0 |
| JSON validity | `json_valid()` | bool |
| Schema compliance | `schema_valid()` | bool |
| Hallucination rate | `hallucination_rate()` | 0.0-1.0 |
| Latency | `Timer` | seconds |
| Input tokens | `count_tokens()` | count |
| Model file size | `os.path.getsize()` | GB |
| Throughput | llama.cpp stats | tok/s |

### 4.1 Derived Analyses (computed in Phase 8)

1. **Quality degradation curve**: F1 vs bits-per-weight, grouped by model family
2. **Pareto frontier**: F1 vs model_size_gb, scatter with annotations
3. **Cliff detection**: Δ(F1) between adjacent quant levels, flag if >5%
4. **Per-field-type F1**: Separate F1 for numeric, string, enum, array fields
5. **Schema complexity interaction**: Simple (5 fields) vs full (8 fields) schemas
6. **Cost comparison**: At what quant does local F1 ≥ cloud baseline F1?

---

## 5. Acceptance Criteria

- [ ] All downloadable GGUF variants present in `models/` (≥20 of 25+)
- [ ] All downloaded variants load and generate output (no crashes)
- [ ] ≥20 variants × 20 pages = 400 inference runs complete (pilot)
- [ ] Key variants × 100 pages complete (full run)
- [ ] Results CSV with all metrics
- [ ] F16 > Q8 ≥ Q6 ≥ Q5 ≥ Q4 > Q3 > Q2 in F1 (approximately monotonic)
- [ ] At least one clear "cliff" identified (>5% F1 drop)
- [ ] No OOM crashes (Mistral-7B Q8 most risky)

---

## 6. Verification

```python
import pandas as pd

df = pd.read_csv("idea-c-quantization/results/experiments.csv")
print(f"Total rows: {len(df)}")
print(f"Model × quant combinations: {df.groupby(['model','quant']).ngroups}")

# Check monotonicity per model
for model in df["model"].unique():
    mdf = df[df["model"] == model]
    quant_order = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
    f1_by_quant = mdf.groupby("quant")["f1"].mean()
    print(f"\n{model}:")
    for q in quant_order:
        if q in f1_by_quant:
            print(f"  {q}: F1={f1_by_quant[q]:.3f}")
```

---

## 7. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Downloads take too long (~80 GB) | Medium | High | Start in Phase 5 background. Prioritize Q4_K_M, Q8_0, Q2_K first |
| Some GGUF filenames differ from guesses | Medium | Medium | Check HuggingFace repo pages manually. Update `DOWNLOADS` dict |
| Mistral-7B Q8 OOM | Medium | Low | Reduce n_ctx to 2048. Or skip Q8 for Mistral |
| F16 3B models slow (~6 GB in memory) | Low | Low | Expected and acceptable. Record latency |
| Q2_K produces garbage output | High | Low | Expected for some models. Record it as data point |
| Disk space insufficient | Medium | High | Check `df -h` before downloading. Need ~80 GB free |

---

## 8. Output Files

```
idea-c-quantization/
├── src/
│   ├── download_models.py      # Download script
│   ├── inference.py             # Unified inference
│   ├── run_experiments.py       # Experiment runner
│   └── run_baselines.py         # (reuses shared/baselines.py)
├── results/
│   └── experiments.csv          # All results
└── figures/                     # Generated in Phase 8

models/
├── qwen2.5-3b/
│   ├── f16.gguf        (~6.0 GB)
│   ├── q8_0.gguf       (~3.2 GB)
│   ├── q6_k.gguf       (~2.5 GB)
│   ├── q5_k_m.gguf     (~2.2 GB)
│   ├── q4_k_m.gguf     (~2.0 GB)   ← already downloaded
│   ├── q3_k_m.gguf     (~1.5 GB)
│   └── q2_k.gguf       (~1.2 GB)
├── llama-3.2-3b/       (same structure)
├── phi-3.5-mini/       (same structure)
└── mistral-7b/
    ├── q8_0.gguf       (~7.5 GB)
    ├── q6_k.gguf       (~5.8 GB)
    ├── q5_k_m.gguf     (~5.0 GB)
    ├── q4_k_m.gguf     (~4.4 GB)
    ├── q3_k_m.gguf     (~3.3 GB)
    └── q2_k.gguf       (~2.5 GB)
```

---

## 9. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Verify/fix download script filenames | 20 min |
| Download GGUF variants (background, started in Phase 5) | 30-60 min |
| Implement inference.py | 20 min |
| Implement run_experiments.py | 20 min |
| Run pilot (20 pages × 25 variants) | 1.5-2 hours |
| Run full experiments (100 pages × key variants) | 1.5-2 hours |
| **Total** | **~4-5 hours** |

**Critical**: Start downloads early (in Phase 5 background). Download priority:
1. Q4_K_M for all models (baseline, ~15 GB total)
2. Q2_K and Q8_0 for all models (extremes, ~25 GB)
3. Remaining variants (~40 GB)
