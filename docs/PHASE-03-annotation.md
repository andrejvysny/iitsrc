# PHASE 3: Auto-Annotation Pipeline

**Duration**: ~2-3 hours
**Schedule**: Day 1, 13:00-16:30
**Dependencies**: Phase 1 (APIs), Phase 2 (HTML pages collected)
**Blocks**: Phase 5, Phase 6, Phase 7, Phase 8 (all need ground truth)

---

## 1. Objective

Create ground truth JSON annotations for all 100 HTML pages using a multi-LLM consensus pipeline. Three annotator models vote per-field; disagreements escalated to GPT-4o. Validated against schema + rules + source grounding. Human review for flagged items.

---

## 2. Pipeline Architecture

```
                            ┌─────────────────────┐
                            │   HTML Page (raw)    │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │ trafilatura.extract  │
                            │ (clean text output)  │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
           ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
           │ Claude Haiku  │  │ GPT-4o-mini  │  │ Gemini Flash │
           │  (Anthropic)  │  │  (OpenAI)    │  │  (CLI/API)   │
           │   temp=0      │  │  temp=0      │  │   temp=0     │
           └───────┬──────┘  └───────┬──────┘  └───────┬──────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  Majority Vote      │
                            │  (per-field, 2/3)   │
                            └──────────┬──────────┘
                                       │
                              ┌────────┴────────┐
                              │  Disagreement?  │
                              └───┬─────────┬───┘
                                  │ yes     │ no
                        ┌─────────▼──┐  ┌───▼───────────┐
                        │ GPT-4o     │  │ Accept        │
                        │ Escalation │  │ consensus     │
                        └─────────┬──┘  └───┬───────────┘
                                  │         │
                                  └────┬────┘
                                       │
                            ┌──────────▼──────────┐
                            │   Validation        │
                            │ • Schema check      │
                            │ • Rule-based check  │
                            │ • Source grounding   │
                            └──────────┬──────────┘
                                       │
                              ┌────────┴────────┐
                              │  Passes all?    │
                              └───┬─────────┬───┘
                                  │ no      │ yes
                        ┌─────────▼──┐  ┌───▼───────────┐
                        │ Flag for   │  │ Save as       │
                        │ human rev  │  │ ground truth  │
                        └────────────┘  └───────────────┘
```

---

## 3. Tasks

### 3.1 Implement `shared/preprocessing.py`

```python
"""HTML preprocessing: clean text extraction + format conversion."""

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify
import json
import re


def html_to_clean_text(html: str) -> str:
    """Extract clean text from HTML using trafilatura."""
    text = trafilatura.extract(html, include_tables=True, include_links=False)
    if not text or len(text) < 50:
        # Fallback: BeautifulSoup get_text
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n", strip=True)
    return text


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown."""
    return markdownify(html, strip=["script", "style", "nav", "footer"])


def html_to_flat_json(html: str) -> str:
    """Convert HTML to flat JSON (sequential text blocks)."""
    soup = BeautifulSoup(html, "lxml")
    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    blocks = {}
    for i, element in enumerate(soup.find_all(text=True)):
        text = element.strip()
        if text and len(text) > 2:
            blocks[f"text_{i}"] = text

    return json.dumps(blocks, ensure_ascii=False)
```

### 3.2 Implement `shared/annotator.py`

```python
"""Multi-LLM annotation pipeline for ground truth generation."""

import json
import os
import time
import subprocess
from pathlib import Path
from typing import Optional

import litellm
from shared.preprocessing import html_to_clean_text
from shared.schemas import get_schema, get_schema_description
from shared.prompts import build_messages
from shared.utils import parse_json_response


class Annotator:
    """Multi-LLM annotation with consensus voting."""

    def __init__(self, domain: str):
        self.domain = domain
        self.schema = get_schema(domain)
        self.schema_str = get_schema_description(domain)

    def annotate_with_claude(self, content: str) -> dict:
        """Annotate using Claude Haiku 4.5."""
        messages = build_messages(self.schema, content)
        response = litellm.completion(
            model="claude-3-5-haiku-20241022",
            messages=messages,
            max_tokens=1024,
            temperature=0,
        )
        return parse_json_response(response.choices[0].message.content)

    def annotate_with_gpt4o_mini(self, content: str) -> dict:
        """Annotate using GPT-4o-mini."""
        messages = build_messages(self.schema, content)
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return parse_json_response(response.choices[0].message.content)

    def annotate_with_gemini(self, content: str) -> dict:
        """Annotate using Gemini 2.0 Flash via CLI or API."""
        # Try API first
        try:
            messages = build_messages(self.schema, content)
            response = litellm.completion(
                model="gemini/gemini-2.0-flash",
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception:
            # Fallback: CLI
            prompt = f"Extract data from this web content as JSON matching this schema:\n{self.schema_str}\n\nContent:\n{content}\n\nOutput ONLY valid JSON:"
            result = subprocess.run(
                ["gemini", "-p", prompt],
                capture_output=True, text=True, timeout=60
            )
            return parse_json_response(result.stdout)

    def escalate_to_gpt4o(self, content: str, field: str, candidates: list) -> any:
        """Escalate disagreement to GPT-4o for single field."""
        prompt = (
            f"Three models extracted different values for '{field}' from this content.\n"
            f"Candidates: {json.dumps(candidates)}\n\n"
            f"Content:\n{content[:2000]}\n\n"
            f"What is the correct value for '{field}'? Output ONLY the value as JSON."
        )
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        return parse_json_response(response.choices[0].message.content)

    def majority_vote(self, results: list[dict]) -> tuple[dict, dict]:
        """Field-level majority vote. Returns (consensus, agreement_scores)."""
        all_fields = set()
        for r in results:
            if r:
                all_fields.update(r.keys())

        consensus = {}
        agreement = {}
        disagreements = []

        for field in all_fields:
            values = [r.get(field) for r in results if r]
            # Count agreements
            from collections import Counter
            # Normalize for comparison
            normalized = []
            for v in values:
                if isinstance(v, str):
                    normalized.append(v.strip().lower())
                else:
                    normalized.append(str(v) if v is not None else "null")

            counts = Counter(normalized)
            most_common_norm, count = counts.most_common(1)[0]

            if count >= 2:  # 2/3 agree
                # Find original value matching normalized
                for v, n in zip(values, normalized):
                    if n == most_common_norm:
                        consensus[field] = v
                        break
                agreement[field] = count / len(values)
            else:
                disagreements.append((field, values))
                agreement[field] = count / len(values)

        return consensus, agreement, disagreements

    def validate(self, data: dict, source_text: str) -> dict:
        """Validate annotation against schema + rules + source."""
        import jsonschema

        result = {"schema_valid": True, "rules_passed": True, "source_grounded": True, "issues": []}

        # Schema validation
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            result["schema_valid"] = False
            result["issues"].append(f"Schema: {e.message}")

        # Rule-based validation
        if "price" in data and data["price"] is not None:
            if not isinstance(data["price"], (int, float)) or data["price"] <= 0:
                result["rules_passed"] = False
                result["issues"].append(f"Invalid price: {data['price']}")

        if "address" in data and data["address"] is not None:
            if len(str(data["address"])) < 10:
                result["rules_passed"] = False
                result["issues"].append(f"Address too short: {data['address']}")

        if "rating" in data and data["rating"] is not None:
            if not (0 <= data["rating"] <= 5):
                result["rules_passed"] = False
                result["issues"].append(f"Rating out of range: {data['rating']}")

        # Source grounding (spot check string values)
        source_lower = source_text.lower() if source_text else ""
        ungrounded = []
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 5:
                if v.lower()[:30] not in source_lower:
                    ungrounded.append(k)
        if len(ungrounded) > len(data) * 0.3:
            result["source_grounded"] = False
            result["issues"].append(f"Ungrounded fields: {ungrounded}")

        return result

    def annotate_page(self, html: str, page_name: str) -> dict:
        """Full annotation pipeline for one page."""
        content = html_to_clean_text(html)

        # Run 3 annotators
        results = []
        per_model = {}

        for name, func in [
            ("claude", self.annotate_with_claude),
            ("gpt4o_mini", self.annotate_with_gpt4o_mini),
            ("gemini", self.annotate_with_gemini),
        ]:
            try:
                r = func(content)
                results.append(r)
                per_model[name] = r
            except Exception as e:
                print(f"  {name} failed for {page_name}: {e}")
                results.append({})
                per_model[name] = {"error": str(e)}

        # Consensus
        consensus, agreement, disagreements = self.majority_vote(results)

        # Escalate disagreements
        escalated_fields = []
        for field, candidates in disagreements:
            try:
                resolved = self.escalate_to_gpt4o(content, field, candidates)
                consensus[field] = resolved
                escalated_fields.append(field)
            except Exception as e:
                print(f"  Escalation failed for {field}: {e}")

        # Validate
        validation = self.validate(consensus, content)

        return {
            "data": consensus,
            "domain": self.domain,
            "page_name": page_name,
            "agreement_scores": agreement,
            "per_model": per_model,
            "escalated_fields": escalated_fields,
            "validation": validation,
            "source_text_length": len(content),
        }
```

### 3.3 Run Annotation Pipeline

```python
# run_annotation.py
from pathlib import Path
import json
from tqdm import tqdm
from shared.annotator import Annotator

data_dir = Path("idea-b-schema-pruning/data/raw_html")
ann_dir = Path("idea-b-schema-pruning/data/annotations")
ann_dir.mkdir(exist_ok=True)

files = sorted(data_dir.glob("*.html"))
print(f"Annotating {len(files)} pages...")

stats = {"total": 0, "success": 0, "flagged": 0, "errors": 0}

for f in tqdm(files):
    # Determine domain from filename
    domain = "ecommerce" if f.name.startswith("ecom_") else "realestate"
    annotator = Annotator(domain)

    try:
        html = f.read_text(encoding="utf-8", errors="replace")
        result = annotator.annotate_page(html, f.stem)

        # Save annotation
        out_path = ann_dir / f"{f.stem}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        # Track stats
        stats["total"] += 1
        if result["validation"]["schema_valid"] and result["validation"]["rules_passed"]:
            stats["success"] += 1
        else:
            stats["flagged"] += 1
            print(f"  FLAGGED: {f.name} - {result['validation']['issues']}")
    except Exception as e:
        stats["errors"] += 1
        print(f"  ERROR: {f.name} - {e}")

print(f"\nAnnotation complete: {stats}")
```

### 3.4 Human Review of Flagged Pages

For each flagged page:
1. Open HTML in browser to see original page
2. Open annotation JSON
3. Check flagged issues
4. Manually correct values if needed
5. Mark as reviewed

```python
# review_flagged.py
import json
from pathlib import Path

ann_dir = Path("idea-b-schema-pruning/data/annotations")

flagged = []
for f in sorted(ann_dir.glob("*.json")):
    ann = json.loads(f.read_text())
    if not ann["validation"]["schema_valid"] or not ann["validation"]["rules_passed"]:
        flagged.append((f.name, ann["validation"]["issues"]))

print(f"Flagged for review: {len(flagged)}")
for name, issues in flagged:
    print(f"  {name}: {issues}")
```

---

## 4. Annotator Model Details

| Model | API | Cost per 100 pages | JSON Mode | Notes |
|-------|-----|---------------------|-----------|-------|
| Claude Haiku 4.5 | Anthropic API | ~$0.15 | Prompt-based | Fast, cheap, good at structured output |
| GPT-4o-mini | OpenAI API | ~$0.10 | `response_format=json_object` | Native JSON mode, very reliable |
| Gemini 2.0 Flash | Gemini CLI / API | $0 (free tier) | Prompt-based | Free but may have rate limits |
| GPT-4o (escalation) | OpenAI API | ~$0.10 (10 pages) | `response_format=json_object` | Only for disagreements (~10% of fields) |

**Total estimated cost**: $0.50-3.00 for 100 pages

---

## 5. Consensus Algorithm Detail

```python
# Per-field majority vote:
# For each field F across 3 model outputs:
#   1. Collect values: [claude[F], gpt[F], gemini[F]]
#   2. Normalize for comparison (lowercase, strip, type coerce)
#   3. If 2/3 agree → accept majority value
#   4. If all 3 disagree → escalate to GPT-4o
#   5. If field present in <2 outputs → use available value if validated
#
# Special handling:
#   - Numeric: within 1% = agreement
#   - String: fuzzy match ratio > 0.9 = agreement
#   - null vs missing: treated as same (field not found)
#   - Arrays (specs): compare as sets of key-value pairs
```

---

## 6. Acceptance Criteria

- [ ] All 100 pages have annotation JSON files in `idea-b-schema-pruning/data/annotations/`
- [ ] All annotations contain: `data`, `domain`, `agreement_scores`, `per_model`, `validation`
- [ ] All annotations pass JSON schema validation (after fixes)
- [ ] All annotations pass rule-based validation (price>0, etc.)
- [ ] Agreement rate ≥ 80% across fields (majority of fields have 2/3 agreement)
- [ ] Flagged pages (<15%) manually reviewed and corrected
- [ ] Each annotation's `data` field has at least 5 non-null values
- [ ] Source grounding: >90% of extracted values found in source text
- [ ] Total API cost < $5

---

## 7. Verification Script

```python
# verify_annotations.py
import json
import jsonschema
from pathlib import Path
from shared.schemas import get_schema

ann_dir = Path("idea-b-schema-pruning/data/annotations")
annotations = sorted(ann_dir.glob("*.json"))
print(f"Total annotations: {len(annotations)}")

stats = {
    "total": len(annotations),
    "schema_valid": 0,
    "rules_valid": 0,
    "grounded": 0,
    "avg_agreement": 0,
    "fields_per_page": [],
}

for f in annotations:
    ann = json.loads(f.read_text())
    domain = ann["domain"]
    schema = get_schema(domain)

    # Schema check
    try:
        jsonschema.validate(ann["data"], schema)
        stats["schema_valid"] += 1
    except jsonschema.ValidationError:
        pass

    # Rule check
    if ann["validation"]["rules_passed"]:
        stats["rules_valid"] += 1

    # Grounding check
    if ann["validation"]["source_grounded"]:
        stats["grounded"] += 1

    # Agreement
    if ann["agreement_scores"]:
        avg = sum(ann["agreement_scores"].values()) / len(ann["agreement_scores"])
        stats["avg_agreement"] += avg

    # Fields count
    non_null = sum(1 for v in ann["data"].values() if v is not None)
    stats["fields_per_page"].append(non_null)

stats["avg_agreement"] /= len(annotations) if annotations else 1
avg_fields = sum(stats["fields_per_page"]) / len(stats["fields_per_page"]) if stats["fields_per_page"] else 0

print(f"\nSchema valid: {stats['schema_valid']}/{stats['total']}")
print(f"Rules valid: {stats['rules_valid']}/{stats['total']}")
print(f"Source grounded: {stats['grounded']}/{stats['total']}")
print(f"Avg agreement: {stats['avg_agreement']:.2%}")
print(f"Avg fields per page: {avg_fields:.1f}")
```

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gemini CLI rate limiting | Medium | Low | Add `time.sleep(1)` between calls. Fallback: 2-model consensus (Claude+GPT) |
| Low agreement on Slovak content | Medium | Medium | Use explicit multilingual system prompt: "Extract data from this content. The content may be in Slovak." GPT-4o escalation handles Slovak well |
| trafilatura strips relevant content | Medium | Medium | Compare with `markdownify` output. Use BeautifulSoup `get_text()` as fallback |
| API errors / timeouts | Low | Low | Add retry with exponential backoff (3 retries, 2s/4s/8s) |
| Budget overrun from escalations | Low | Low | Cap escalations at 15 pages. Accept majority if 2/3 agree on most fields |
| Annotation quality too low | Low | High | Spot-check 10 random annotations against manual inspection. If <80% accuracy, consider annotating manually |

---

## 9. Output Files

```
idea-b-schema-pruning/data/annotations/
├── ecom_amazon_001.json
├── ecom_amazon_002.json
├── ...
├── realestate_reality_013.json
└── annotation_stats.json          # Overall statistics
```

Each annotation JSON structure:
```json
{
  "data": {
    "name": "...",
    "price": 29.99,
    "currency": "USD",
    "brand": "...",
    "description": "...",
    "rating": 4.5,
    "availability": "in_stock",
    "specs": [{"key": "...", "value": "..."}]
  },
  "domain": "ecommerce",
  "page_name": "ecom_amazon_001",
  "agreement_scores": {
    "name": 1.0,
    "price": 1.0,
    "currency": 1.0,
    "brand": 0.67,
    "description": 0.67,
    "rating": 1.0,
    "availability": 1.0,
    "specs": 0.67
  },
  "per_model": {
    "claude": { ... },
    "gpt4o_mini": { ... },
    "gemini": { ... }
  },
  "escalated_fields": ["brand"],
  "validation": {
    "schema_valid": true,
    "rules_passed": true,
    "source_grounded": true,
    "issues": []
  },
  "source_text_length": 2847
}
```

---

## 10. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Implement preprocessing.py | 20 min |
| Implement annotator.py | 40 min |
| Run annotation (100 pages × 3 models) | 30-45 min |
| Implement consensus + validation | 20 min |
| Review + fix flagged pages | 30-45 min |
| Verify annotations | 15 min |
| **Total** | **~2.5-3 hours** |
