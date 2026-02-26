# PHASE 5: Idea B — Schema-Aware Pruning Implementation

**Duration**: ~3-4 hours
**Schedule**: Day 2, 08:00-12:00
**Dependencies**: Phase 1 (models), Phase 2 (data), Phase 3 (annotations), Phase 4 (metrics/prompts)
**Blocks**: Phase 8 (analysis)

---

## 1. Objective

Implement 4 DOM pruning strategies (none, generic, schema-heuristic, schema-semantic), 3 output format converters (HTML, Markdown, flat JSON), and the local LLM extraction pipeline. Run all 36 configurations (4×3×3) on 100 pages.

---

## 2. Tasks

### 2.1 Implement `idea-b-schema-pruning/src/pruning.py`

```python
"""Four DOM pruning strategies for schema-conditioned extraction."""

from bs4 import BeautifulSoup, Comment, NavigableString
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from shared.schemas import SCHEMA_KEYWORDS, get_schema_field_names, get_schema


# ── Strategy 1: No Pruning ──────────────────────────────────────────────

def no_pruning(html: str) -> str:
    """Return raw HTML as-is."""
    return html


# ── Strategy 2: Generic Pruning ─────────────────────────────────────────

REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside",
               "iframe", "noscript", "svg", "link", "meta"}

REMOVE_ID_CLASS_PATTERNS = re.compile(
    r'cookie|consent|gdpr|banner|advert|sponsor|promo|popup|modal|overlay',
    re.IGNORECASE
)

REMOVE_ATTRS = {"style", "onclick", "onload", "onerror", "onmouseover",
                "data-tracking", "data-analytics"}


def generic_pruning(html: str) -> str:
    """Remove non-content elements: scripts, styles, nav, footer, ads, etc."""
    soup = BeautifulSoup(html, "lxml")

    # Remove tag types
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove elements with cookie/ad class/id patterns
    for el in soup.find_all(True):
        el_id = el.get("id", "") or ""
        el_class = " ".join(el.get("class", []))
        if REMOVE_ID_CLASS_PATTERNS.search(el_id) or REMOVE_ID_CLASS_PATTERNS.search(el_class):
            el.decompose()
            continue

        # Remove unnecessary attributes
        attrs_to_remove = []
        for attr in el.attrs:
            if attr in REMOVE_ATTRS or attr.startswith("data-"):
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del el[attr]

    return str(soup)


# ── Strategy 3: Schema-Aware Heuristic Pruning ──────────────────────────

def _get_text_tokens(text: str) -> set[str]:
    """Tokenize text into lowercase words."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def schema_heuristic_pruning(html: str, domain: str, threshold: float = 0.05) -> str:
    """Prune DOM based on keyword overlap with schema fields.

    Args:
        html: Raw or generically pruned HTML
        domain: 'ecommerce' or 'realestate'
        threshold: Minimum keyword overlap score to keep a subtree
    """
    # First apply generic pruning
    soup = BeautifulSoup(generic_pruning(html), "lxml")
    keywords = SCHEMA_KEYWORDS[domain]

    def score_node(node) -> float:
        text = node.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            return 0.0
        tokens = _get_text_tokens(text)
        overlap = tokens & keywords
        return len(overlap) / len(keywords)

    # Walk tree and remove low-score subtrees
    body = soup.find("body") or soup
    _prune_recursive(body, keywords, threshold)

    return str(soup)


def _prune_recursive(node, keywords: set, threshold: float):
    """Recursively prune subtrees with low keyword overlap."""
    if isinstance(node, NavigableString):
        return

    children = list(node.children)
    for child in children:
        if isinstance(child, NavigableString):
            continue
        if not hasattr(child, 'get_text'):
            continue

        text = child.get_text(separator=" ", strip=True)
        if not text:
            child.decompose()
            continue

        tokens = _get_text_tokens(text)
        overlap = tokens & keywords
        score = len(overlap) / len(keywords) if keywords else 0

        if score < threshold and len(text) < 200:
            # Low relevance, short text → remove
            child.decompose()
        else:
            # Recurse into children
            _prune_recursive(child, keywords, threshold)


# ── Strategy 4: Schema-Aware Semantic Pruning ────────────────────────────

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_field_descriptions(domain: str) -> list[str]:
    """Get human-readable descriptions for each schema field."""
    schema = get_schema(domain)
    descriptions = []
    for field_name, field_def in schema["properties"].items():
        desc = field_def.get("description", field_name)
        descriptions.append(f"{field_name}: {desc}")
    return descriptions


def schema_semantic_pruning(html: str, domain: str, threshold: float = 0.25) -> str:
    """Prune DOM based on embedding similarity with schema fields.

    Args:
        html: Raw or generically pruned HTML
        domain: 'ecommerce' or 'realestate'
        threshold: Minimum cosine similarity to keep a node
    """
    # First apply generic pruning
    cleaned = generic_pruning(html)
    soup = BeautifulSoup(cleaned, "lxml")

    model = _get_embedding_model()

    # Embed schema field descriptions (cached per domain)
    field_descs = _get_field_descriptions(domain)
    field_embeddings = model.encode(field_descs)

    # Collect text nodes with their DOM elements
    body = soup.find("body") or soup
    nodes_to_check = []

    for el in body.find_all(True):
        # Get direct text content (not children's text)
        direct_text = el.find(string=True, recursive=False)
        full_text = el.get_text(separator=" ", strip=True)
        text = full_text if full_text else ""

        if text and len(text) > 5:
            nodes_to_check.append((el, text))

    if not nodes_to_check:
        return str(soup)

    # Batch embed all node texts
    texts = [t for _, t in nodes_to_check]
    text_embeddings = model.encode(texts, batch_size=64)

    # Compute similarities
    sims = cosine_similarity(text_embeddings, field_embeddings)  # shape: (n_nodes, n_fields)
    max_sims = sims.max(axis=1)  # max similarity to any field per node

    # Mark nodes to keep
    keep_elements = set()
    for (el, _), max_sim in zip(nodes_to_check, max_sims):
        if max_sim >= threshold:
            # Keep this element and all its ancestors
            keep_elements.add(id(el))
            parent = el.parent
            while parent:
                keep_elements.add(id(parent))
                parent = parent.parent

    # Remove nodes not in keep set (bottom-up to avoid modifying tree during traversal)
    for el in reversed(list(body.find_all(True))):
        if id(el) not in keep_elements:
            # Only decompose leaf-level unneeded nodes
            if not any(id(child) in keep_elements for child in el.find_all(True)):
                el.decompose()

    return str(soup)


# ── Dispatcher ───────────────────────────────────────────────────────────

STRATEGIES = {
    "none": no_pruning,
    "generic": generic_pruning,
    "heuristic": schema_heuristic_pruning,
    "semantic": schema_semantic_pruning,
}

def apply_pruning(html: str, strategy: str, domain: str = None, **kwargs) -> str:
    """Apply a pruning strategy by name."""
    if strategy == "none":
        return no_pruning(html)
    elif strategy == "generic":
        return generic_pruning(html)
    elif strategy == "heuristic":
        return schema_heuristic_pruning(html, domain, **kwargs)
    elif strategy == "semantic":
        return schema_semantic_pruning(html, domain, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

### 2.2 Implement Output Format Converters

Add to `shared/preprocessing.py` or `idea-b-schema-pruning/src/pruning.py`:

```python
from markdownify import markdownify as md_convert
import json
from bs4 import BeautifulSoup


def to_simplified_html(pruned_html: str) -> str:
    """Return pruned HTML as-is (simplified by pruning)."""
    return pruned_html


def to_markdown(pruned_html: str) -> str:
    """Convert pruned HTML to Markdown."""
    return md_convert(pruned_html, strip=["img"]).strip()


def to_flat_json(pruned_html: str) -> str:
    """Convert pruned HTML to flat JSON (sequential text blocks)."""
    soup = BeautifulSoup(pruned_html, "lxml")
    blocks = {}
    idx = 0
    for element in soup.find_all(string=True):
        text = element.strip()
        if text and len(text) > 1:
            blocks[f"text_{idx}"] = text
            idx += 1
    return json.dumps(blocks, ensure_ascii=False, indent=None)


FORMAT_CONVERTERS = {
    "html": to_simplified_html,
    "markdown": to_markdown,
    "flat_json": to_flat_json,
}

def convert_format(pruned_html: str, fmt: str) -> str:
    """Convert pruned HTML to specified format."""
    return FORMAT_CONVERTERS[fmt](pruned_html)
```

### 2.3 Implement `idea-b-schema-pruning/src/extract.py`

```python
"""Local LLM extraction pipeline using llama-cpp-python."""

from llama_cpp import Llama
from pathlib import Path
import json
import time

from shared.prompts import build_extraction_prompt
from shared.utils import parse_json_response, Timer, count_tokens
from shared.schemas import get_schema


# Model configs
MODEL_CONFIGS = {
    "qwen2.5-3b": {
        "path": "models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
        "n_ctx": 4096,
    },
    "llama-3.2-3b": {
        "path": "models/llama-3.2-3b/llama-3.2-3b-instruct-q4_k_m.gguf",
        "chat_format": "llama-3",
        "n_ctx": 4096,
    },
    "phi-3.5-mini": {
        "path": "models/phi-3.5-mini/phi-3.5-mini-instruct-q4_k_m.gguf",
        "chat_format": "phi-3",
        "n_ctx": 4096,
    },
}

_loaded_models = {}


def load_model(model_name: str, model_path: str = None, n_ctx: int = 4096) -> Llama:
    """Load a GGUF model with Metal acceleration. Caches loaded models."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    config = MODEL_CONFIGS.get(model_name, {})
    path = model_path or config.get("path")

    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")

    llm = Llama(
        model_path=str(path),
        n_gpu_layers=-1,       # Full Metal offload
        n_ctx=n_ctx or config.get("n_ctx", 4096),
        verbose=False,
        chat_format=config.get("chat_format"),
    )

    _loaded_models[model_name] = llm
    return llm


def extract_with_local_model(
    content: str,
    domain: str,
    model_name: str,
    model_path: str = None,
    max_tokens: int = 1024,
) -> dict:
    """Run extraction on a local GGUF model.

    Returns:
        dict with: raw_output, parsed, latency_s, tokens_in, tokens_out, model
    """
    schema = get_schema(domain)
    llm = load_model(model_name, model_path)

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

    tokens_in = count_tokens(content)

    with Timer() as t:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            # Note: JSON grammar can be added here for constrained output
        )

    raw_output = response["choices"][0]["message"]["content"]
    parsed = parse_json_response(raw_output)

    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "latency_s": t.elapsed,
        "tokens_in": tokens_in,
        "tokens_out": response["usage"]["completion_tokens"] if "usage" in response else None,
        "model": model_name,
    }
```

### 2.4 Implement `idea-b-schema-pruning/src/run_experiments.py`

```python
"""Run all Idea B experiments: 4 pruning × 3 formats × 3 models × 100 pages."""

import json
import csv
from pathlib import Path
from tqdm import tqdm

from shared.schemas import get_schema
from shared.metrics import compute_all_metrics
from shared.utils import determine_domain

from pruning import apply_pruning, convert_format
from extract import extract_with_local_model, MODEL_CONFIGS


DATA_DIR = Path("idea-b-schema-pruning/data")
RESULTS_DIR = Path("idea-b-schema-pruning/results")
RESULTS_DIR.mkdir(exist_ok=True)

PRUNING_STRATEGIES = ["none", "generic", "heuristic", "semantic"]
OUTPUT_FORMATS = ["html", "markdown", "flat_json"]
MODELS = list(MODEL_CONFIGS.keys())


def load_pages() -> list[dict]:
    """Load all HTML pages and their annotations."""
    pages = []
    html_dir = DATA_DIR / "raw_html"
    ann_dir = DATA_DIR / "annotations"

    for html_file in sorted(html_dir.glob("*.html")):
        ann_file = ann_dir / f"{html_file.stem}.json"
        if not ann_file.exists():
            print(f"WARNING: No annotation for {html_file.name}, skipping")
            continue

        pages.append({
            "name": html_file.stem,
            "html": html_file.read_text(encoding="utf-8", errors="replace"),
            "annotation": json.loads(ann_file.read_text()),
            "domain": determine_domain(html_file.name),
        })

    return pages


def run_single_experiment(page: dict, strategy: str, fmt: str, model: str) -> dict:
    """Run one configuration on one page."""
    # Prune
    pruned = apply_pruning(page["html"], strategy, domain=page["domain"])

    # Convert format
    content = convert_format(pruned, fmt)

    # Extract
    result = extract_with_local_model(content, page["domain"], model)

    # Compute metrics
    gold = page["annotation"]["data"]
    schema = get_schema(page["domain"])
    source_text = page["html"]  # Use original HTML for hallucination check
    metrics = compute_all_metrics(
        result["parsed"], gold, schema, source_text, result["raw_output"]
    )

    return {
        "page_id": page["name"],
        "domain": page["domain"],
        "pruning": strategy,
        "format": fmt,
        "model": model,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "exact_match": metrics["exact_match"],
        "json_valid": metrics["json_valid"],
        "schema_valid": metrics["schema_valid"],
        "hallucination_rate": metrics["hallucination_rate"],
        "tokens_in": result["tokens_in"],
        "latency_s": result["latency_s"],
        "raw_output": result["raw_output"],
    }


def run_all():
    """Run full experiment matrix."""
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")

    results = []
    total = len(PRUNING_STRATEGIES) * len(OUTPUT_FORMATS) * len(MODELS) * len(pages)

    with open(RESULTS_DIR / "experiments.csv", "w", newline="") as f:
        writer = None

        for model in MODELS:
            print(f"\n=== Model: {model} ===")
            for strategy in PRUNING_STRATEGIES:
                for fmt in OUTPUT_FORMATS:
                    print(f"  {strategy}/{fmt}...")
                    for page in tqdm(pages, desc=f"{strategy}/{fmt}/{model}"):
                        try:
                            row = run_single_experiment(page, strategy, fmt, model)
                            results.append(row)

                            if writer is None:
                                fieldnames = [k for k in row.keys() if k != "raw_output"]
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()

                            writer.writerow({k: v for k, v in row.items() if k != "raw_output"})
                            f.flush()
                        except Exception as e:
                            print(f"    ERROR: {page['name']} - {e}")

    print(f"\nDone! {len(results)} experiments saved to {RESULTS_DIR / 'experiments.csv'}")


if __name__ == "__main__":
    run_all()
```

### 2.5 Implement `idea-b-schema-pruning/src/run_baselines.py`

```python
"""Run cloud baselines for Idea B (best format per pruning strategy)."""

import json
import csv
from pathlib import Path
from tqdm import tqdm

from shared.schemas import get_schema
from shared.metrics import compute_all_metrics
from shared.baselines import CLOUD_MODELS, run_cloud_model
from shared.utils import determine_domain
from shared.preprocessing import html_to_clean_text

from pruning import apply_pruning, convert_format


DATA_DIR = Path("idea-b-schema-pruning/data")
RESULTS_DIR = Path("idea-b-schema-pruning/results")

# Use best format (determined after local experiments, default to flat_json)
BEST_FORMAT = "flat_json"
PRUNING_STRATEGIES = ["none", "generic", "heuristic", "semantic"]


def run_cloud_baselines():
    """Run all cloud models on all pruning strategies."""
    html_dir = DATA_DIR / "raw_html"
    ann_dir = DATA_DIR / "annotations"

    pages = []
    for html_file in sorted(html_dir.glob("*.html")):
        ann_file = ann_dir / f"{html_file.stem}.json"
        if ann_file.exists():
            pages.append({
                "name": html_file.stem,
                "html": html_file.read_text(encoding="utf-8", errors="replace"),
                "annotation": json.loads(ann_file.read_text()),
                "domain": determine_domain(html_file.name),
            })

    results = []
    for model_name, model_id in CLOUD_MODELS.items():
        for strategy in PRUNING_STRATEGIES:
            print(f"  {model_name} / {strategy}...")
            for page in tqdm(pages, desc=f"{model_name}/{strategy}"):
                try:
                    pruned = apply_pruning(page["html"], strategy, domain=page["domain"])
                    content = convert_format(pruned, BEST_FORMAT)

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
                        "format": BEST_FORMAT,
                        "model": model_name,
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "json_valid": metrics["json_valid"],
                        "schema_valid": metrics["schema_valid"],
                        "tokens_in": result.get("input_tokens"),
                        "latency_s": result.get("latency_s"),
                    })
                except Exception as e:
                    print(f"    ERROR: {page['name']} - {e}")

    # Save
    with open(RESULTS_DIR / "cloud_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} cloud baseline results")


if __name__ == "__main__":
    run_cloud_baselines()
```

---

## 3. Experiment Matrix Detail

### 3.1 Local Experiments: 36 Configurations

| Config ID | Pruning | Format | Model |
|-----------|---------|--------|-------|
| L01 | none | html | qwen2.5-3b |
| L02 | none | markdown | qwen2.5-3b |
| L03 | none | flat_json | qwen2.5-3b |
| L04 | generic | html | qwen2.5-3b |
| L05 | generic | markdown | qwen2.5-3b |
| L06 | generic | flat_json | qwen2.5-3b |
| L07 | heuristic | html | qwen2.5-3b |
| L08 | heuristic | markdown | qwen2.5-3b |
| L09 | heuristic | flat_json | qwen2.5-3b |
| L10 | semantic | html | qwen2.5-3b |
| L11 | semantic | markdown | qwen2.5-3b |
| L12 | semantic | flat_json | qwen2.5-3b |
| L13-L24 | same pattern | | llama-3.2-3b |
| L25-L36 | same pattern | | phi-3.5-mini |

### 3.2 Cloud Baselines: 20 Configurations

| Config ID | Pruning | Model |
|-----------|---------|-------|
| C01-C04 | none/generic/heur/sem | gpt-4o |
| C05-C08 | same | claude-sonnet |
| C09-C12 | same | qwen-72b |
| C13-C16 | same | llama-70b |
| C17-C20 | same | mistral-large |

### 3.3 Time Estimation

**Per local experiment (1 config × 100 pages)**:
- No pruning: ~15s/page × 100 = 25 min (large input, slow)
- Generic: ~10s/page × 100 = 17 min
- Heuristic: ~8s/page × 100 = 13 min
- Semantic: ~8s/page × 100 = 13 min (+ 2 min embedding overhead)

**Total for 36 configs**: ~36 × 17 min avg = ~10 hours

**Time reduction strategies**:
1. Run on 50 pages instead of 100 for non-critical configs → 5 hours
2. Skip "none" pruning for non-Qwen models (known to fail) → 30 configs, ~8 hours
3. Run only best format per pruning after initial pilot (9 pages) → 12 configs, ~3.5 hours
4. **Recommended**: Run pilot (10 pages) for all 36 configs, then full run for top 12 → ~4 hours total

### 3.4 Preprocessing Cache

Pruning is fast (~0.1s/page), inference is slow (~10s/page). Cache pruned+formatted content:

```python
CACHE_DIR = DATA_DIR / "processed"
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_content(page_name: str, strategy: str, fmt: str) -> str | None:
    cache_file = CACHE_DIR / f"{page_name}_{strategy}_{fmt}.txt"
    if cache_file.exists():
        return cache_file.read_text()
    return None

def cache_content(page_name: str, strategy: str, fmt: str, content: str):
    cache_file = CACHE_DIR / f"{page_name}_{strategy}_{fmt}.txt"
    cache_file.write_text(content)
```

---

## 4. Acceptance Criteria

- [ ] All 4 pruning strategies produce valid output for all 100 pages (no crashes)
- [ ] Token reduction measurements:
  - Generic: ≥60% reduction vs raw HTML
  - Heuristic: ≥80% reduction
  - Semantic: ≥90% reduction
- [ ] All 3 output format converters produce non-empty output
- [ ] At least 12 experiment configs complete on full 100 pages
- [ ] Results CSV contains all required columns
- [ ] No OOM errors during inference
- [ ] Semantic pruning embedding model loads and runs on CPU

---

## 5. Verification

```python
import pandas as pd

df = pd.read_csv("idea-b-schema-pruning/results/experiments.csv")
print(f"Total rows: {len(df)}")
print(f"Configs tested: {df.groupby(['pruning','format','model']).ngroups}")

# Average F1 by pruning strategy
print("\nAvg F1 by pruning:")
print(df.groupby("pruning")["f1"].mean().sort_values(ascending=False))

# Average token count by pruning
print("\nAvg tokens by pruning:")
print(df.groupby("pruning")["tokens_in"].mean().sort_values())

# JSON validity by model
print("\nJSON validity by model:")
print(df.groupby("model")["json_valid"].mean())
```

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Raw HTML too large for context window | High | Medium | Truncate to n_ctx (4096 tokens). "none" strategy expected to fail for large pages |
| Semantic pruning slow (embedding 100s of nodes) | Medium | Low | Batch encoding. Only embed nodes with >5 chars text |
| sentence-transformers OOM | Low | Medium | all-MiniLM-L6-v2 is only 22M params, should be fine |
| heuristic threshold too aggressive | Medium | Medium | Start with 0.05, tune on 20 validation pages |
| Experiments take >6 hours | High | High | Use time reduction strategies (pilot + top configs) |
| Model cache between configs incorrect | Low | Medium | Clear `_loaded_models` between model switches |

---

## 7. Output Files

```
idea-b-schema-pruning/
├── src/
│   ├── pruning.py              # 4 strategies + 3 format converters
│   ├── extract.py              # Local LLM extraction
│   ├── run_experiments.py      # Experiment runner
│   └── run_baselines.py        # Cloud baseline runner
├── data/
│   └── processed/              # Cached pruned content (optional)
├── results/
│   ├── experiments.csv         # Main results (36 configs × 100 pages)
│   └── cloud_baselines.csv     # Cloud results (20 configs × 100 pages)
└── figures/                    # Generated in Phase 8
```

---

## 8. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Implement pruning.py (4 strategies) | 45 min |
| Implement format converters | 15 min |
| Implement extract.py | 20 min |
| Implement run_experiments.py | 20 min |
| Implement run_baselines.py | 15 min |
| Download remaining 2 models (Llama, Phi) | 10 min (parallel) |
| Run pilot (10 pages × 36 configs) | 30-45 min |
| Run full experiments (100 pages × top configs) | 2-3 hours |
| **Total** | **~4-5 hours** |
