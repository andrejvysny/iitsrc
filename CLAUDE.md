# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: optimizing small local LLMs (<7B params, GGUF) for structured web data extraction (e-commerce + real estate). Two parallel research tracks share dataset, schemas, metrics, and infrastructure:

- **Idea B** (`idea-b-schema-pruning/`): Schema-aware DOM pruning — 4 pruning strategies × 3 output formats × 3 models
- **Idea C** (`idea-c-quantization/`): Quantization impact study — 4 models × 7 quant levels (F16→Q2)

Output: IEEE 6-page conference paper for IIT.SRC 2026. Deadline: Feb 28, 2026.

Hardware: MacBook Pro M4 Pro, 24GB unified RAM, Metal GPU acceleration.

## Commands

```bash
source .venv/bin/activate

pip install -r requirements.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# verify setup
./verify_setup.sh

# tests (when test files exist)
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_metrics.py::TestFieldF1::test_perfect_match -v

# experiments (when implemented)
python idea-b-schema-pruning/src/run_experiments.py
python idea-c-quantization/src/run_experiments.py
python idea-b-schema-pruning/src/run_baselines.py

# compile paper
cd paper-b && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Shared infrastructure (`shared/`) — IMPLEMENTED

- `schemas.py` — Two fixed JSON schemas (ecommerce: 8 fields, realestate: 8 fields) + keyword sets for heuristic pruning. Entry point: `get_schema(domain)` where domain is `"ecommerce"` or `"realestate"`
- `metrics.py` — Primary metric: field-level macro F1 via `field_f1()`. String matching uses rapidfuzz ≥0.85, numbers within 1% tolerance, lists use intersection/gold ≥0.5. `compute_all_metrics()` returns f1, precision, recall, exact_match, json_valid, schema_valid, hallucination_rate
- `prompts.py` — `build_extraction_prompt()` for local models (single string), `build_messages()` for chat APIs
- `utils.py` — `parse_json_safe()` (handles code blocks, brace extraction), `timer()` context manager, `count_tokens()` via tiktoken

### Shared — PLANNED (not yet implemented)

- `baselines.py` — Cloud API wrappers (GPT-4o, Claude Sonnet, Qwen-72B, Llama-70B, Mistral-Large via litellm)
- `annotator.py` — Ground truth generation: 3 LLMs → majority vote → GPT-4o escalation
- `preprocessing.py` — HTML cleaning + format converters (simplified HTML, markdown via markdownify, flat JSON)

### Idea B (`idea-b-schema-pruning/src/`) — PLANNED

- `pruning.py` — 4 strategies: no pruning → generic (remove scripts/nav/ads) → schema-heuristic (keyword overlap) → schema-semantic (embedding similarity via all-MiniLM-L6-v2)
- `extract.py` — Local GGUF inference via llama-cpp-python with Metal
- `run_experiments.py` — 36 local + 20 cloud configs across 100 pages

### Idea C (`idea-c-quantization/src/`) — PLANNED

- `download_models.py` — Fetch all GGUF variants from HuggingFace
- `inference.py` — Unified inference across quant levels
- `run_experiments.py` — 25 model variants across 100 pages

### Data (`idea-b-schema-pruning/data/`, idea-c symlinks to it)

- `raw_html/` — 100 HTML pages (50 ecom from Amazon/eBay/Alza/Mall, 50 realestate from Zillow/Realtor/Nehnutelnosti/Reality)
- `annotations/` — Ground truth JSONs
- `processed/` — Cached pruned outputs
- File naming: `{domain}_{site}_{id}.html` (e.g., `ecom_amazon_001.html`)

### Models (`models/`)

- 4 families: Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini, Mistral-7B
- 7 quant levels each: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K
- Mistral-7B F16 won't fit in 24GB RAM — cap at Q5_K_M

### Papers (`paper-b/`, `paper-c/`)

- IEEE format LaTeX, `figures/` subdirs for charts

## Key conventions

- Two domains only: `"ecommerce"` and `"realestate"` — passed as string to `get_schema(domain)`
- All LLM calls use temperature=0, max_tokens=1024
- Local inference: llama-cpp-python with Metal GPU, JSON grammar constraint
- API calls: litellm for OpenRouter models, native SDKs for OpenAI/Anthropic
- Environment variables in `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` (loaded via python-dotenv)
- Python 3.14, follow PEP 8
- Phased implementation: `docs/PHASE-{01..10}-*.md`, full spec in `docs/SPECIFICATION.md`
