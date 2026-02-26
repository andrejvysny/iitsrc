# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: optimizing small local LLMs (<7B params, GGUF) for structured web data extraction (e-commerce + real estate). Two parallel research tracks share dataset, schemas, metrics, and infrastructure:

- **Idea B** (`idea-b-schema-pruning/`): Schema-aware DOM pruning — 4 pruning strategies × 3 output formats × 3 models
- **Idea C** (`idea-c-quantization/`): Quantization impact study — 4 models × 7 quant levels (F16→Q2)

Output: IEEE 6-page conference paper for IIT.SRC 2026.

## Commands

```bash
# activate venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# run tests
python -m pytest tests/test_metrics.py -v

# run single test
python -m pytest tests/test_metrics.py::TestFieldF1::test_perfect_match -v

# experiment runners (when implemented)
python idea-b-schema-pruning/src/run_experiments.py
python idea-c-quantization/src/run_experiments.py
python idea-b-schema-pruning/src/run_baselines.py

# compile paper
cd paper-b && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Shared infrastructure (`shared/`)
- `schemas.py` — Two fixed JSON schemas (ecommerce: 8 fields, realestate: 8 fields) + keyword sets for heuristic pruning
- `metrics.py` — Primary metric: field-level macro F1. Also: precision, recall, exact match, JSON validity, schema compliance, hallucination rate. String matching uses rapidfuzz ≥0.85, numbers within 1% tolerance, lists use intersection/gold ≥0.5
- `prompts.py` — System prompt + extraction template. Two builders: `build_extraction_prompt()` for local models (single string), `build_messages()` for chat APIs
- `baselines.py` — Cloud API wrappers (GPT-4o, Claude Sonnet, Qwen-72B, Llama-70B, Mistral-Large via litellm)
- `annotator.py` — Ground truth generation: 3 LLMs in parallel → majority vote → GPT-4o escalation → validation
- `preprocessing.py` — HTML cleaning + format converters (simplified HTML, markdown via markdownify, flat JSON)

### Idea B (`idea-b-schema-pruning/src/`)
- `pruning.py` — 4 strategies: no pruning → generic (remove scripts/nav/ads) → schema-heuristic (keyword overlap) → schema-semantic (embedding similarity via all-MiniLM-L6-v2)
- `extract.py` — Local GGUF inference via llama-cpp-python with Metal
- `run_experiments.py` — 36 local + 20 cloud configs across 100 pages

### Idea C (`idea-c-quantization/src/`)
- `download_models.py` — Fetch all GGUF variants from HuggingFace
- `inference.py` — Unified inference across quant levels
- `run_experiments.py` — 25 model variants across 100 pages

### Data (`idea-b-schema-pruning/data/`, symlinked from idea-c)
- `raw_html/` — 100 HTML pages (50 ecom from Amazon/eBay/Alza/Mall, 50 realestate from Zillow/Realtor/Nehnutelnosti/Reality)
- `annotations/` — Ground truth JSONs
- `processed/` — Cached pruned outputs
- File naming: `{domain}_{site}_{id}.html` (e.g., `ecom_amazon_001.html`)

### Models (`models/`)
- 4 families: Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini, Mistral-7B
- 7 quant levels each: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K
- Mistral-7B F16 won't fit in 24GB RAM — cap at Q5_K_M

### Papers (`paper-b/`, `paper-c/`)
- IEEE format LaTeX, `figures/` symlinked from respective idea dirs

## Key conventions

- Two domains only: `"ecommerce"` and `"realestate"` — passed as string to `get_schema(domain)`
- All LLM calls use temperature=0, max_tokens=1024
- Local inference: llama-cpp-python with Metal GPU acceleration, JSON grammar constraint
- API calls: litellm for OpenRouter models, native SDKs for OpenAI/Anthropic
- Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` (loaded via python-dotenv)
- Python 3.14, no linting/formatting config — follow PEP 8
- Phases documented in `docs/PHASE-{01..10}-*.md`, full spec in `docs/SPECIFICATION.md`
