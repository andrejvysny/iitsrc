# PHASE 4: Shared Infrastructure

**Duration**: ~1.5-2 hours
**Schedule**: Day 1, 16:30-20:00
**Dependencies**: Phase 1 (environment), Phase 2 (schemas)
**Blocks**: Phase 5 (pruning uses metrics/prompts/utils), Phase 6 (quantization uses same), Phase 7 (baselines)

---

## 1. Objective

Implement all shared Python modules: metrics, prompts, baselines (cloud API wrappers), utils (JSON parsing, timing, memory). Write unit tests for metrics. Verify cloud baseline calls work end-to-end.

---

## 2. Tasks

### 2.1 Implement `shared/metrics.py`

**Already partially implemented.** Must include:

| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize_string(s)` | `str → str` | Lowercase, strip whitespace |
| `values_match(pred, gold, fuzzy_threshold=0.85)` | `any, any → bool` | Exact + fuzzy matching |
| `field_f1(predicted, gold)` | `dict, dict → dict` | Per-field and macro F1, precision, recall |
| `exact_match(predicted, gold)` | `dict, dict → bool` | All fields match? |
| `json_valid(output)` | `str → bool` | `json.loads()` succeeds? |
| `schema_valid(data, schema)` | `dict, dict → bool` | `jsonschema.validate()` passes? |
| `hallucination_rate(predicted, source_text)` | `dict, str → float` | Fraction of values not in source |
| `compute_all_metrics(predicted, gold, schema, source_text, raw_output)` | `→ dict` | All metrics combined |

**Edge cases to handle**:
- Empty dicts: `field_f1({}, {})` → all zeros
- Null values: `null` in gold and `null` in predicted = TP
- Missing vs null: key absent vs key=null treated differently
- Numeric tolerance: within 1% relative = match
- Fuzzy string: `rapidfuzz.fuzz.ratio > 85` = match
- List fields (specs): intersection over gold ≥ 0.5
- Unicode/Slovak text: normalize properly

### 2.2 Implement `shared/prompts.py`

**Already partially implemented.** Must include:

| Function | Description |
|----------|-------------|
| `SYSTEM_PROMPT` | System message constant |
| `EXTRACTION_TEMPLATE` | Full prompt template with `{system}`, `{schema}`, `{content}` |
| `build_extraction_prompt(schema, content, include_system=True)` | For llama.cpp (single string) |
| `build_messages(schema, content)` | For API models (list of message dicts) |

### 2.3 Implement `shared/baselines.py`

```python
"""Cloud LLM baseline runners."""

import time
import json
import litellm
from shared.prompts import build_messages
from shared.utils import parse_json_response, Timer


def run_cloud_model(content: str, schema: dict, model: str,
                    provider: str = "auto") -> dict:
    """Run extraction on a cloud model. Returns result dict with metrics."""
    messages = build_messages(schema, content)

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0,
    }

    # OpenAI-specific: JSON mode
    if "gpt-" in model:
        kwargs["response_format"] = {"type": "json_object"}

    with Timer() as t:
        response = litellm.completion(**kwargs)

    raw_output = response.choices[0].message.content
    parsed = parse_json_response(raw_output)

    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "latency_s": t.elapsed,
        "model": model,
        "input_tokens": response.usage.prompt_tokens if response.usage else None,
        "output_tokens": response.usage.completion_tokens if response.usage else None,
    }


# Convenience wrappers
CLOUD_MODELS = {
    "gpt-4o": "gpt-4o",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "qwen-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
    "llama-70b": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "mistral-large": "openrouter/mistralai/mistral-large-latest",
}


def run_all_baselines(content: str, schema: dict) -> dict:
    """Run all 5 cloud baselines. Returns dict of model → result."""
    results = {}
    for name, model_id in CLOUD_MODELS.items():
        try:
            results[name] = run_cloud_model(content, schema, model_id)
        except Exception as e:
            results[name] = {"error": str(e), "model": model_id}
    return results
```

### 2.4 Implement `shared/utils.py`

```python
"""Utility functions: JSON parsing, timing, memory measurement."""

import json
import re
import time
import os


class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM output, handling common formatting issues."""
    if not text:
        return {}

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    return {}


def get_model_size_gb(model_path: str) -> float:
    """Get model file size in GB."""
    return os.path.getsize(model_path) / (1024 ** 3)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def truncate_content(content: str, max_tokens: int = 3000) -> str:
    """Truncate content to max_tokens (approximate)."""
    # Rough: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n[TRUNCATED]"


def determine_domain(filename: str) -> str:
    """Determine domain from filename convention."""
    if filename.startswith("ecom_"):
        return "ecommerce"
    elif filename.startswith("realestate_"):
        return "realestate"
    else:
        raise ValueError(f"Cannot determine domain from filename: {filename}")
```

### 2.5 Write Unit Tests for Metrics

```python
# tests/test_metrics.py
import pytest
from shared.metrics import (
    field_f1, exact_match, json_valid, schema_valid,
    hallucination_rate, values_match, compute_all_metrics
)
from shared.schemas import ECOM_SCHEMA


class TestValuesMatch:
    def test_exact_strings(self):
        assert values_match("iPhone 15", "iPhone 15")

    def test_case_insensitive(self):
        assert values_match("Apple", "apple")

    def test_fuzzy_match(self):
        assert values_match("iPhone 15 Pro Max", "iPhone 15 Pro Max 256GB")  # close enough

    def test_numeric_exact(self):
        assert values_match(999, 999)

    def test_numeric_tolerance(self):
        assert values_match(999.0, 999.01)

    def test_null_both(self):
        assert values_match(None, None)

    def test_null_one(self):
        assert not values_match(None, "value")

    def test_list_match(self):
        gold = [{"key": "Color", "value": "Black"}]
        pred = [{"key": "color", "value": "black"}]
        assert values_match(pred, gold)


class TestFieldF1:
    def test_perfect_match(self):
        gold = {"name": "iPhone", "price": 999, "brand": "Apple"}
        pred = {"name": "iPhone", "price": 999, "brand": "Apple"}
        result = field_f1(pred, gold)
        assert result["macro_f1"] == 1.0

    def test_empty_prediction(self):
        gold = {"name": "iPhone", "price": 999}
        pred = {}
        result = field_f1(pred, gold)
        assert result["macro_f1"] == 0.0
        assert result["fn"] == 2

    def test_partial_match(self):
        gold = {"name": "iPhone", "price": 999, "brand": "Apple"}
        pred = {"name": "iPhone", "price": 999, "brand": "Samsung"}
        result = field_f1(pred, gold)
        assert 0 < result["macro_f1"] < 1.0

    def test_extra_fields(self):
        gold = {"name": "iPhone"}
        pred = {"name": "iPhone", "color": "black"}
        result = field_f1(pred, gold)
        assert result["fp"] >= 1


class TestJsonValid:
    def test_valid(self):
        assert json_valid('{"name": "test"}')

    def test_invalid(self):
        assert not json_valid('not json')

    def test_empty(self):
        assert not json_valid('')

    def test_none(self):
        assert not json_valid(None)


class TestSchemaValid:
    def test_valid_ecom(self):
        data = {
            "name": "iPhone", "price": 999, "currency": "USD",
            "description": "A phone", "availability": "in_stock",
        }
        assert schema_valid(data, ECOM_SCHEMA)

    def test_missing_required(self):
        data = {"name": "iPhone"}  # missing price, currency, etc.
        assert not schema_valid(data, ECOM_SCHEMA)

    def test_invalid_enum(self):
        data = {
            "name": "iPhone", "price": 999, "currency": "BTC",
            "description": "A phone", "availability": "in_stock",
        }
        assert not schema_valid(data, ECOM_SCHEMA)


class TestHallucinationRate:
    def test_no_hallucination(self):
        pred = {"name": "iPhone 15", "price": 999}
        source = "Buy the iPhone 15 for only 999 dollars"
        assert hallucination_rate(pred, source) == 0.0

    def test_full_hallucination(self):
        pred = {"name": "Samsung Galaxy", "brand": "Samsung"}
        source = "Buy the iPhone 15 for only 999 dollars"
        rate = hallucination_rate(pred, source)
        assert rate > 0.5

    def test_empty_prediction(self):
        assert hallucination_rate({}, "some text") == 0.0
```

**Run tests**:
```bash
source .venv/bin/activate
python -m pytest tests/test_metrics.py -v
```

---

## 3. Acceptance Criteria

- [ ] `shared/metrics.py` — all functions implemented, handles edge cases
- [ ] `shared/prompts.py` — prompt template generates correct output for both schemas
- [ ] `shared/baselines.py` — `run_cloud_model()` successfully calls all 5 cloud models
- [ ] `shared/utils.py` — JSON parser handles: raw JSON, ```json fenced, JSON in text, broken JSON
- [ ] `tests/test_metrics.py` — all tests pass (10+ test cases)
- [ ] Cloud API test: each of 5 baselines returns valid JSON extraction on a test page

---

## 4. Verification

```python
# Quick end-to-end test
from shared.schemas import get_schema, get_schema_description
from shared.prompts import build_messages, build_extraction_prompt
from shared.metrics import field_f1, compute_all_metrics
from shared.baselines import run_cloud_model
from shared.utils import parse_json_response, Timer

# Test prompts
schema = get_schema("ecommerce")
prompt = build_extraction_prompt(schema, "iPhone 15 Pro, $999, Apple, in stock")
print(f"Prompt length: {len(prompt)} chars")

# Test metrics
gold = {"name": "iPhone 15 Pro", "price": 999, "brand": "Apple"}
pred = {"name": "iPhone 15 Pro", "price": 999, "brand": "apple"}
result = field_f1(pred, gold)
print(f"F1: {result['macro_f1']:.3f}")

# Test JSON parsing
assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
assert parse_json_response('Here is the result: {"a": 1}') == {"a": 1}

# Test cloud baseline (1 call)
result = run_cloud_model("iPhone 15, $999, Apple", schema, "gpt-4o-mini")
print(f"Cloud test: {result['parsed']}")

print("All shared infrastructure tests passed!")
```

---

## 5. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Fuzzy matching too lenient/strict | Medium | Medium | Tune threshold (0.85 default). Test on known pairs |
| API rate limits during baseline test | Low | Low | Test one model at a time with sleep between |
| JSON parser misses edge cases | Medium | Low | Add more patterns as discovered during experiments |
| litellm incompatibility | Low | Medium | Fall back to direct openai/anthropic SDK calls |

---

## 6. Output Files

```
shared/
├── __init__.py
├── schemas.py          # JSON schemas (already exists)
├── metrics.py          # Evaluation metrics (already exists, verify completeness)
├── prompts.py          # Prompt templates (already exists)
├── preprocessing.py    # HTML cleaning (implemented in Phase 3)
├── baselines.py        # Cloud API wrappers
├── annotator.py        # Annotation pipeline (implemented in Phase 3)
└── utils.py            # Helpers

tests/
└── test_metrics.py     # Unit tests
```

---

## 7. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Review + complete metrics.py | 20 min |
| Implement baselines.py | 20 min |
| Implement utils.py | 15 min |
| Write unit tests | 20 min |
| Run tests + fix | 15 min |
| Test cloud baselines (5 API calls) | 10 min |
| End-to-end verification | 10 min |
| **Total** | **~1.5-2 hours** |
