# SPECIFICATION: Optimizing Small LLMs for Structured Web Data Extraction

## 1. Project Overview

**Title**: Optimizing Small Language Models for Structured Data Extraction from Web Sources
**Context**: Diploma thesis at FIIT STU Bratislava
**Output**: IIT.SRC 2026 conference paper (6 pages, IEEE format, English, double-blind)
**Deadline**: February 28, 2026, 23:59 CET (EasyChair submission)
**Timeline**: 3 full working days (Feb 26-28, 8-12h each)

### 1.1 Problem Statement

Structured data extraction from web pages (e-commerce product info, real estate listings, etc.) currently relies on expensive cloud LLMs (GPT-4, Claude). These models are accurate but costly ($6+/1K pages), require internet, and raise data privacy concerns. Small local models (<7B parameters) are 50-100x cheaper and fully private, but produce lower quality extractions without optimization.

### 1.2 Research Goal

Demonstrate that optimized small models (<7B params, quantized, running locally on consumer hardware) can approach frontier LLM extraction quality at a fraction of the cost. Two complementary research angles are investigated; supervisor selects one for final paper submission.

### 1.3 Two Research Ideas

| | Idea B: Schema-Conditioned DOM Pruning | Idea C: Quantization Impact Study |
|---|---|---|
| **RQ** | Can schema-aware HTML preprocessing enable sub-3B models to match frontier LLM extraction quality while reducing input tokens by 90%+? | How does quantization (F16→Q2) affect structured extraction quality, and where is the optimal quality-efficiency tradeoff? |
| **Novelty** | First work conditioning DOM pruning on target extraction schema | First systematic study of quantization impact on schema compliance + field-level F1 for web extraction |
| **Output** | 4 pruning strategies ablated across 3 models × 3 formats | 4 models × 7 quant levels = 28 variants benchmarked |

Both ideas share the same dataset, schemas, metrics, baselines, and infrastructure.

---

## 2. Research Background

### 2.1 State of the Art

| Paper | Venue/Year | Key Finding | Relevance |
|-------|------------|-------------|-----------|
| AXE | arXiv 2025 (2602.01838) | DOM pruning + 0.6B model achieves F1=88.1% on SWDE with 97.9% token reduction | Direct predecessor for Idea B; uses generic pruning, not schema-aware |
| AutoScraper | EMNLP 2024 | GPT-4-Turbo achieves F1=88.69% on SWDE zero-shot | Cloud baseline reference |
| SLOT | EMNLP Industry 2025 (2505.04016) | Fine-tuned Mistral-7B: 99.5% schema accuracy via constrained decoding | Shows fine-tuning potential; we study zero-shot |
| ScrapeGraphAI-100k | arXiv 2025 (2602.15189) | Fine-tuned Qwen3-1.7B competitive with 30B models | Dataset source + shows small model potential |
| NEXT-EVAL | arXiv 2025 (2505.17125) | Flat JSON preprocessing yields F1=0.9567 | Output format inspiration |
| HtmlRAG | WWW 2025 (2411.02959) | HTML > plain text for LLM understanding | Supports preserving HTML structure |
| JSONSchemaBench | arXiv 2025 (2501.10868) | 10K schema benchmark for constrained decoding | Related benchmarking methodology |
| LoRA | ICLR 2022 (2106.09685) | Low-rank adaptation for efficient fine-tuning | Background reference |
| DOM-LM | EMNLP 2022 (2201.10608) | DOM-aware language model pretraining | Structural understanding reference |
| QLoRA | NeurIPS 2023 (2305.14314) | 4-bit quantized fine-tuning | Quantization background for Idea C |
| GoLLIE | ICLR 2024 | Code-based schema IE | Alternative schema representation |

### 2.2 Research Gaps Addressed

1. **Gap for Idea B**: Schema-aware DOM pruning is unexplored. AXE (2025) uses generic pruning (remove scripts/nav/footer). Nobody adapts pruning strategy to the target extraction schema — e.g., keeping price-related nodes when extracting product data but dropping them for article extraction.

2. **Gap for Idea C**: No systematic study of quantization impact specifically on structured output tasks. Existing quantization benchmarks (MMLU, perplexity) measure general capability. Nobody has measured how quantization degrades JSON validity, schema compliance, or per-field-type extraction accuracy.

### 2.3 Must-Cite References (BibTeX keys)

1. `hu2022lora` — LoRA (ICLR 2022)
2. `deng2022domlm` — DOM-LM (EMNLP 2022)
3. `dettmers2023qlora` — QLoRA (NeurIPS 2023)
4. `axe2025` — AXE (arXiv 2025)
5. `scrapegraphai2025` — ScrapeGraphAI-100k (arXiv 2025)
6. `slot2025` — SLOT (EMNLP Industry 2025)
7. `htmlrag2025` — HtmlRAG (WWW 2025)
8. `nexteval2025` — NEXT-EVAL (arXiv 2025)
9. `jsonschemabench2025` — JSONSchemaBench (arXiv 2025)
10. `autoscraper2024` — AutoScraper (EMNLP 2024)
11. `gollie2024` — GoLLIE (ICLR 2024)

---

## 3. Technical Specification

### 3.1 Hardware & Environment

| Component | Value |
|-----------|-------|
| Machine | MacBook Pro M4 Pro |
| RAM | 24 GB unified memory |
| GPU | Apple M4 Pro GPU (Metal acceleration) |
| OS | macOS Darwin 24.6.0 |
| Python | 3.14.0 (via Homebrew) |
| Inference engine | llama-cpp-python with Metal backend |
| Model format | GGUF (llama.cpp native) |
| Max model size | ~14 GB (F16 of 7B) — tight fit in 24 GB |

### 3.2 API Access & Budget

| Service | Access | Estimated Cost |
|---------|--------|----------------|
| OpenAI API | API key (OPENAI_API_KEY env var) | $5-15 |
| Anthropic API | API key (ANTHROPIC_API_KEY env var) | $3-10 |
| OpenRouter | API key (OPENROUTER_API_KEY env var) | $5-15 |
| Gemini CLI | Free tier (google-gemini CLI) | $0 |
| **Total budget** | | **$10-50** |

### 3.3 Domains & Schemas

#### E-Commerce Schema (8 fields)

```json
{
  "type": "object",
  "properties": {
    "name":         {"type": "string", "description": "Product name/title"},
    "price":        {"type": "number", "description": "Product price as a number"},
    "currency":     {"type": "string", "enum": ["USD","EUR","CZK","GBP"]},
    "brand":        {"type": ["string","null"], "description": "Brand/manufacturer"},
    "description":  {"type": "string", "description": "Product description"},
    "rating":       {"type": ["number","null"], "minimum": 0, "maximum": 5},
    "availability": {"type": "string", "enum": ["in_stock","out_of_stock","unknown"]},
    "specs":        {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key","value"]}}
  },
  "required": ["name","price","currency","description","availability"]
}
```

**Field types**: 3 string, 1 number, 1 enum-string, 1 nullable-string, 1 nullable-number, 1 array-of-objects
**Complexity**: Medium (mixed types, nested array)

#### Real Estate Schema (8 fields)

```json
{
  "type": "object",
  "properties": {
    "address":    {"type": "string", "description": "Full property address"},
    "price":      {"type": "number", "description": "Listing price"},
    "currency":   {"type": "string", "enum": ["USD","EUR","GBP"]},
    "bedrooms":   {"type": ["integer","null"]},
    "bathrooms":  {"type": ["integer","null"]},
    "area_sqm":   {"type": ["number","null"], "description": "Area in sq meters"},
    "description":{"type": "string"},
    "type":       {"type": "string", "enum": ["apartment","house","condo","land","unknown"]}
  },
  "required": ["address","price","currency","description","type"]
}
```

**Field types**: 2 string, 1 number, 2 enum-string, 2 nullable-integer, 1 nullable-number
**Complexity**: Medium (multiple nullable numeric fields)

### 3.4 Dataset Specification

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Custom HTML pages | Manual collection from 8 websites | 100 pages (50 ecom + 50 realestate) | Primary evaluation set |
| ScrapeGraphAI-100k | HuggingFace (`Ilanit/ScrapeGraphAI_100k`) | 500 sampled examples | Extended evaluation |

#### Custom Pages Breakdown

| Domain | Site | Count | Language |
|--------|------|-------|----------|
| E-commerce | Amazon.com | 12-13 | English |
| E-commerce | eBay.com | 12-13 | English |
| E-commerce | Alza.sk | 12-13 | Slovak |
| E-commerce | Mall.sk | 12-13 | Slovak |
| Real estate | Zillow.com | 12-13 | English |
| Real estate | Realtor.com | 12-13 | English |
| Real estate | Nehnutelnosti.sk | 12-13 | Slovak |
| Real estate | Reality.sk | 12-13 | Slovak |

**File naming**: `{domain}_{site}_{id}.html` (e.g., `ecom_amazon_001.html`, `realestate_zillow_001.html`)
**Split**: 80 test + 20 validation (for threshold tuning in pruning strategies)
**Collection method**: Browser save (Cmd+S) to capture rendered DOM including JS-generated content
**Encoding**: UTF-8

#### ScrapeGraphAI-100k Sampling

- Source: HuggingFace dataset `Ilanit/ScrapeGraphAI_100k`
- Sample: 500 examples with diverse schemas (stratified by schema complexity)
- Used for: Extended evaluation to show generalization beyond our 2 domains

### 3.5 Models

#### Local Models (GGUF, llama-cpp-python with Metal)

| Model | HuggingFace ID | Params | Q4_K_M Size | Context | Chat Template |
|-------|---------------|--------|-------------|---------|---------------|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct-GGUF` | 3B | ~2.0 GB | 32K | ChatML |
| Llama-3.2-3B-Instruct | `bartowski/Llama-3.2-3B-Instruct-GGUF` (or official) | 3B | ~2.0 GB | 128K | Llama-3 |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF` (or official) | 3.8B | ~2.4 GB | 128K | Phi-3 |
| Mistral-7B-Instruct-v0.3 | `TheBloke/Mistral-7B-Instruct-v0.3-GGUF` (or official) | 7.3B | ~4.4 GB | 32K | Mistral |

**Idea B uses**: Qwen-3B, Llama-3B, Phi-3.5-mini (all Q4_K_M)
**Idea C uses**: All 4 models × 7 quantization levels

#### Quantization Levels (Idea C)

| Level | Bits/Weight | Typical 3B Size | Typical 7B Size | Description |
|-------|-------------|-----------------|-----------------|-------------|
| F16 | 16 | ~6.0 GB | ~14.0 GB | Full precision baseline |
| Q8_0 | 8 | ~3.2 GB | ~7.5 GB | Near-lossless |
| Q6_K | 6 | ~2.5 GB | ~5.8 GB | High quality |
| Q5_K_M | 5 | ~2.2 GB | ~5.0 GB | Good quality |
| Q4_K_M | 4 | ~2.0 GB | ~4.4 GB | Standard tradeoff |
| Q3_K_M | 3 | ~1.5 GB | ~3.3 GB | Aggressive compression |
| Q2_K | 2 | ~1.2 GB | ~2.5 GB | Extreme, expect degradation |

**Note**: Mistral-7B F16 (~14 GB) may not load with 24 GB RAM (model + KV cache + OS). Limited to Q5_K_M and below.

#### Cloud Baselines (API)

| Model | Provider | API Model ID | ~Cost/100 pages |
|-------|----------|-------------|-----------------|
| GPT-4o | OpenAI | `gpt-4o` | ~$0.60 |
| Claude 3.5 Sonnet | Anthropic | `claude-3-5-sonnet-20241022` | ~$0.60 |
| Qwen2.5-72B-Instruct | OpenRouter | `qwen/qwen-2.5-72b-instruct` | ~$0.20 |
| Llama-3.1-70B-Instruct | OpenRouter | `meta-llama/llama-3.1-70b-instruct` | ~$0.20 |
| Mistral-Large | OpenRouter | `mistralai/mistral-large-latest` | ~$0.20 |

### 3.6 Prompt Design

**System prompt** (used for all models):
```
You are a structured data extraction assistant. Extract data from web content
according to the JSON schema below. Output ONLY valid JSON. Use null for missing fields.
Do not invent values not present in the content.
```

**User prompt template**:
```
JSON Schema:
{schema_json}

Web Content:
{preprocessed_content}

Output:
```

**Parameters**:
- Temperature: 0 (deterministic)
- Max tokens: 1024 (sufficient for any single extraction)
- For local models: JSON grammar constraint via llama.cpp grammar feature
- For OpenAI: `response_format={"type": "json_object"}`
- For Anthropic: prompt-based JSON enforcement

### 3.7 Metrics

#### Quality Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Field-level F1** (primary) | Standard F1 over fields. TP=field present+value matches, FP=wrong/extra, FN=missing | Per-field and macro-averaged |
| **Precision** | TP/(TP+FP) | Fraction of extracted fields that are correct |
| **Recall** | TP/(TP+FN) | Fraction of gold fields successfully extracted |
| **Exact Match** | All fields match = 1, else 0 | Binary per-page metric |
| **JSON Validity** | `json.loads()` succeeds | % of outputs that are valid JSON |
| **Schema Compliance** | `jsonschema.validate()` passes | % of valid JSON outputs matching schema |
| **Hallucination Rate** | Fraction of predicted values not found in source text | Measures fabrication |

**String matching**: Exact match first, then fuzzy match (rapidfuzz ratio ≥ 0.85)
**Numeric matching**: Within 1% relative tolerance
**List matching**: Intersection over gold set ≥ 0.5

#### Efficiency Metrics

| Metric | Measurement | Unit |
|--------|-------------|------|
| Latency | `time.perf_counter()` around inference | seconds |
| Throughput | llama.cpp reported tokens/second | tok/s |
| Peak Memory | Model file size + runtime estimate | GB |
| Input Tokens | `tiktoken` encoding length | tokens |
| Cost | API pricing or amortized hardware | $/page |

### 3.8 Auto-Annotation Pipeline

```
HTML page
  → trafilatura.extract(html)     # clean text extraction
  → 3 LLMs in parallel:
      1. Claude Haiku 4.5 (Anthropic API, temp=0)
      2. GPT-4o-mini (OpenAI API, temp=0, json mode)
      3. Gemini 2.0 Flash (gemini CLI, free tier)
  → Field-level majority vote (2/3 agree = accept)
  → Disagreements → GPT-4o escalation ($0.01/page)
  → Validation:
      - jsonschema compliance
      - Rule-based: price>0, address.length>10, 0≤rating≤5
      - Source grounding: extracted value found in source text
  → Human review of flagged items (~5-15 pages)
  → Final ground truth JSON
```

**Estimated annotation cost**: $1-3 for 100 pages
**Output per page**: `{page_name}.json` containing:
```json
{
  "data": { ... },           // consensus extraction
  "domain": "ecommerce",     // or "realestate"
  "agreement_scores": { "field": 0.67|1.0, ... },
  "per_model": {
    "claude": { ... },
    "gpt4o_mini": { ... },
    "gemini": { ... }
  },
  "escalated_fields": ["field1"],
  "validation": {
    "schema_valid": true,
    "rules_passed": true,
    "source_grounded": true
  }
}
```

---

## 4. Idea B: Schema-Conditioned DOM Pruning — Detailed Design

### 4.1 Four Pruning Strategies

#### Strategy 1: No Pruning (baseline)
- Input: Raw HTML as-is
- Expected tokens: ~10K-30K per page
- Purpose: Baseline to show that raw HTML overwhelms small models

#### Strategy 2: Generic Pruning
- Remove elements: `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<iframe>`, `<noscript>`, `<svg>`
- Remove: HTML comments (`<!-- ... -->`)
- Remove: `data-*` attributes, `style` attributes, `class` attributes, `onclick`/event handlers
- Remove: Cookie banners (heuristic: elements with id/class matching `cookie|consent|gdpr|banner`)
- Remove: Ad containers (heuristic: elements with id/class matching `ad|advertisement|sponsor|promo`)
- Preserve: All text content, links, tables, lists, headings, images (alt text only)
- Expected tokens: ~2K-5K per page
- Implementation: BeautifulSoup tree walk + tag/attribute filtering

#### Strategy 3: Schema-Aware Heuristic Pruning
- Step 1: Apply generic pruning
- Step 2: Tokenize each DOM subtree's text content → set of lowercase tokens
- Step 3: Compute keyword overlap score with schema:
  ```
  score(node) = |tokens(node.text) ∩ schema_keywords| / |schema_keywords|
  ```
- Step 4: Keep subtrees with `score > threshold` (threshold tuned on 20-page validation set)
- Schema keywords per domain defined in `shared/schemas.py` (see `ECOM_KEYWORDS`, `REALESTATE_KEYWORDS`)
- Expected tokens: ~500-2K per page
- Implementation: BeautifulSoup tree walk + token set intersection

#### Strategy 4: Schema-Aware Semantic Pruning
- Step 1: Apply generic pruning
- Step 2: Load `all-MiniLM-L6-v2` sentence transformer (22M params, fast on CPU)
- Step 3: Embed each DOM node's text content (batched)
- Step 4: Embed each schema field name + description (cached, computed once per schema)
- Step 5: Compute cosine similarity between each node embedding and all field embeddings
- Step 6: Keep nodes where `max(similarities) > threshold` (threshold tuned on validation set)
- Expected tokens: ~300-1K per page (most aggressive, highest quality)
- Implementation: sentence-transformers + sklearn cosine_similarity

### 4.2 Output Format Converters

| Format | Method | Preserves |
|--------|--------|-----------|
| Simplified HTML | Pruned DOM serialized as HTML string | Tags, structure, tables |
| Markdown | `markdownify.markdownify(pruned_html)` | Headers, lists, tables, emphasis |
| Flat JSON | Sequential text blocks: `{"text_1": "...", "text_2": "...", ...}` | Text content only, ordered |

### 4.3 Experiment Matrix

```
4 pruning strategies × 3 output formats × 3 local models = 36 local configs
5 cloud baselines × 4 pruning strategies (best format only) = 20 cloud configs
Total: 56 configurations
Each config tested on: 100 custom pages
Selected configs also on: 500 ScrapeGraphAI pages
```

### 4.4 Expected Results Hypothesis

| Pruning | Best Model | Expected F1 | Expected Token Reduction |
|---------|-----------|-------------|-------------------------|
| None | Qwen-3B | 0.30-0.50 | 0% |
| Generic | Qwen-3B | 0.55-0.70 | 70% |
| Schema-heuristic | Qwen-3B | 0.70-0.82 | 90% |
| Schema-semantic | Qwen-3B | 0.75-0.88 | 95% |
| Schema-semantic | GPT-4o | 0.88-0.95 | 95% |

---

## 5. Idea C: Quantization Impact Study — Detailed Design

### 5.1 Experiment Matrix

```
3 small models × 7 quant levels = 21 variants
1 larger model (Mistral-7B) × 4 quant levels = 4 variants (F16/Q8 may not fit)
Total: 25 model variants
Each tested on: 100 custom pages (best preprocessing from Idea B)
Selected configs on: 500 ScrapeGraphAI pages
+ 5 cloud baselines for comparison
```

### 5.2 Key Analyses

1. **Quality Degradation Curve**: Line chart, X=bits/weight, Y=F1, one line per model family
2. **Pareto Frontier**: Scatter plot, X=peak memory (GB), Y=F1, annotated with model+quant
3. **Cliff Detection**: Identify quant level where F1 drops >5% vs previous level
4. **Schema Complexity Interaction**: Heatmap, rows=quant levels, cols=schema complexity bins
5. **Per-Field-Type Analysis**: Grouped bar chart, F1 for numeric/text/enum/list fields across quant levels
6. **Cost-Quality Comparison**: At what quant level does local model match cheapest cloud baseline?

### 5.3 Hypotheses

- Q4_K_M is the sweet spot (general NLP literature consensus)
- Numeric fields (price, bedrooms) more robust to quantization than free-text (description)
- Complex schemas (10+ fields) degrade faster under aggressive quantization
- Q3_K_M+ viable for simple schemas (3-5 fields)
- All Q4+ local models cheaper than cloud baselines by 50-100x

---

## 6. Project Directory Structure

```
iitsrc/
├── docs/                           # THIS DIRECTORY
│   ├── SPECIFICATION.md            # This file
│   ├── PHASE-01-setup.md
│   ├── PHASE-02-data.md
│   ├── PHASE-03-annotation.md
│   ├── PHASE-04-shared.md
│   ├── PHASE-05-pruning.md
│   ├── PHASE-06-quantization.md
│   ├── PHASE-07-baselines.md
│   ├── PHASE-08-analysis.md
│   ├── PHASE-09-paper.md
│   └── PHASE-10-submission.md
│
├── shared/                         # Shared code for both ideas
│   ├── __init__.py
│   ├── schemas.py                  # JSON schemas + field keywords
│   ├── metrics.py                  # F1, precision, recall, validity, hallucination
│   ├── prompts.py                  # Prompt templates
│   ├── preprocessing.py            # HTML cleaning, format conversion
│   ├── annotator.py                # Multi-LLM annotation pipeline
│   ├── baselines.py                # Cloud API calls
│   └── utils.py                    # Helpers (timing, memory, JSON parsing)
│
├── idea-b-schema-pruning/          # Idea B: Schema-Conditioned DOM Pruning
│   ├── src/
│   │   ├── pruning.py              # 4 pruning strategies
│   │   ├── extract.py              # Local LLM extraction pipeline
│   │   ├── run_experiments.py      # Full experiment runner
│   │   └── run_baselines.py        # Cloud baseline runner
│   ├── data/
│   │   ├── raw_html/               # Original HTML pages (100 files)
│   │   ├── annotations/            # Ground truth JSONs (100 files)
│   │   └── processed/              # Pruned HTML outputs (cached)
│   ├── results/                    # Experiment CSVs
│   ├── notebooks/
│   │   └── analysis.ipynb          # Results visualization
│   └── figures/                    # Generated charts (PDF)
│
├── idea-c-quantization/            # Idea C: Quantization Study
│   ├── src/
│   │   ├── download_models.py      # Download all GGUF variants
│   │   ├── inference.py            # Unified inference pipeline
│   │   ├── run_experiments.py      # Full experiment runner
│   │   └── run_baselines.py        # Cloud baseline runner
│   ├── data/                       # Symlinked to idea-b data
│   ├── results/                    # Experiment CSVs
│   ├── notebooks/
│   │   └── analysis.ipynb          # Pareto charts, degradation curves
│   └── figures/                    # Generated charts (PDF)
│
├── models/                         # Downloaded GGUF files
│   ├── qwen2.5-3b/                 # All quant variants
│   ├── llama-3.2-3b/
│   ├── phi-3.5-mini/
│   └── mistral-7b/
│
├── paper-b/                        # LaTeX for Idea B paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/                    # Symlinked from idea-b/figures
│
├── paper-c/                        # LaTeX for Idea C paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/                    # Symlinked from idea-c/figures
│
├── ieee_latex_template-1/          # Original IEEE template (reference)
├── .venv/                          # Python virtual environment
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 7. Paper Structure (IEEE 6-page, double-blind)

### Paper B: Schema-Conditioned DOM Pruning

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | Problem, schema-aware pruning approach, key F1 result, practical takeaway |
| 1. Introduction | 0.75 | Motivation (cloud LLM cost), gap (no schema-aware pruning), 3-4 contribution bullets, teaser figure |
| 2. Related Work | 0.75 | LLM-based extraction (AutoScraper, AXE), HTML preprocessing (HtmlRAG, NEXT-EVAL), structured output (SLOT, JSONSchemaBench) |
| 3. Method | 1.5 | Pipeline architecture diagram, 4 pruning strategies with formulas, output formats, prompt design |
| 4. Experiments | 1.5 | Setup (models, data, metrics), main results table, ablation (pruning × format), cloud comparison, per-domain analysis |
| 5. Discussion & Conclusion | 0.75 | Key findings, limitations (zero-shot only, single-page), future work (learned pruner, fine-tuning) |
| References | 0.5 | 15-20 references |

**Key figures**: (1) Pipeline architecture diagram, (2) F1 comparison bar chart across pruning strategies, (3) Token reduction vs F1 scatter plot

### Paper C: Quantization Impact Study

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | Problem, systematic quantization study scope, key sweet-spot finding, practical recommendation |
| 1. Introduction | 0.75 | Motivation (edge deployment, privacy), gap (no extraction-specific quant benchmarks), contributions |
| 2. Related Work | 0.75 | Quantization methods (GPTQ, AWQ, GGUF), LLM extraction, structured output constraints |
| 3. Method | 1.5 | Experimental design, models × quant matrix, preprocessing, metrics, evaluation protocol |
| 4. Experiments | 1.5 | Degradation curves, Pareto frontier, cliff analysis, per-field-type breakdown, cloud comparison |
| 5. Discussion & Conclusion | 0.75 | Sweet spot recommendation, per-field insights, limitations, future work |
| References | 0.5 | 15-20 references |

**Key figures**: (1) Quality degradation curve, (2) Pareto frontier (memory vs F1), (3) Per-field-type heatmap

---

## 8. Timeline Summary

| Time | Day 1 (Feb 26) | Day 2 (Feb 27) | Day 3 (Feb 28) |
|------|-----------------|-----------------|-----------------|
| 08-10 | Phase 1: Setup | Phase 5: Pruning impl | Phase 9: Paper writing |
| 10-12 | Phase 2: Data collection | Phase 5: Run experiments | Phase 9: Paper writing |
| 12-13 | Lunch | Lunch | Lunch |
| 13-15 | Phase 3: Annotation | Phase 6: Quant experiments | Phase 9: Paper writing |
| 15-17 | Phase 3: Review | Phase 6: Run experiments | Phase 9: Figures + final |
| 17-18 | Phase 4: Shared code | Phase 7: Cloud baselines | Phase 9: Proofread |
| 18-20 | Phase 4: Test + verify | Phase 8: Analysis + charts | Phase 10: Submit |
| 20-22 | Download models (bg) | Phase 8: Finalize | Buffer / emergency |

---

## 9. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| llama-cpp-python Metal compile fail | High | Low | Use pre-built wheel or mlx-lm fallback |
| Model download slow/interrupted | Medium | Medium | Start downloads early, work in parallel |
| Websites block scraping | Medium | Medium | Browser save (Cmd+S), curl with user-agent |
| Slovak sites encoding issues | Low | Medium | UTF-8 enforcement, test parsing early |
| JS-rendered content missing in wget | Medium | High | Save from browser (captures rendered DOM) |
| Gemini CLI rate limiting | Low | Medium | 2-model consensus fallback, sleep between calls |
| Low annotation agreement on Slovak text | Medium | Medium | Multilingual prompt, GPT-4o escalation |
| Experiments take too long | High | Medium | Reduce to subset (50 pages), skip lowest-priority configs |
| Mistral-7B F16 OOM | Low | High | Cap at Q5_K_M for 7B model |
| API budget exceeded | Medium | Low | Track costs per call, cap at $50 |
| EasyChair slow near deadline | Medium | Medium | Submit by 22:00, not 23:59 |

---

## 10. Success Criteria

### Minimum Viable Paper
- [ ] 100 HTML pages collected and annotated
- [ ] At least 12 local model configurations tested (2 pruning × 2 formats × 3 models, or 3 models × 4 quants)
- [ ] At least 2 cloud baselines (GPT-4o + Claude Sonnet)
- [ ] 3+ publication-quality figures
- [ ] Complete 6-page IEEE paper submitted before deadline

### Target Results
- [ ] All 36 (Idea B) or 25 (Idea C) configurations tested on 100 pages
- [ ] All 5 cloud baselines tested
- [ ] ScrapeGraphAI-100k extended evaluation (500 pages)
- [ ] Clear finding: best local config achieves ≥80% of GPT-4o F1
- [ ] Statistical summary with mean ± std across pages

### Stretch Goals
- [ ] Both papers written (supervisor picks one)
- [ ] Combined analysis across both ideas
- [ ] ScrapeGraphAI-100k full evaluation
