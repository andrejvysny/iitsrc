# PHASE 1: Environment & Infrastructure Setup

**Duration**: ~2 hours
**Schedule**: Day 1, 08:00-10:00
**Dependencies**: None (first phase)
**Blocks**: All subsequent phases

---

## 1. Objective

Set up a fully working development environment with all tools installed and verified: Python venv, all dependencies, project directory structure, API keys, first model downloaded and tested with Metal GPU acceleration.

---

## 2. Tasks

### 2.1 Create Python Virtual Environment

```bash
cd /Users/andrejvysny/fiit/dp/iitsrc
python3 -m venv .venv
source .venv/bin/activate
```

**Note**: Python 3.14.0 is available via Homebrew at `/opt/homebrew/bin/python3`.

### 2.2 Install Dependencies

**requirements.txt** (complete list):
```
# LLM inference
llama-cpp-python          # Local GGUF model inference with Metal
litellm                   # Unified API for cloud LLMs
openai                    # OpenAI API client
anthropic                 # Anthropic API client

# HTML processing
trafilatura               # HTML-to-text extraction
beautifulsoup4            # DOM parsing and manipulation
lxml                      # Fast HTML/XML parser
markdownify               # HTML-to-Markdown conversion
html5lib                  # Tolerant HTML parser

# Embeddings (schema-semantic pruning)
sentence-transformers     # all-MiniLM-L6-v2 for semantic similarity

# Data & ML
scikit-learn              # cosine_similarity, metrics
pandas                    # DataFrames for results
numpy                     # Numerical operations

# Visualization
matplotlib                # Charts and plots
seaborn                   # Statistical visualization

# Schema validation
jsonschema                # JSON Schema validation

# Dataset
datasets                  # HuggingFace datasets library
huggingface-hub           # Model/dataset downloads

# Utilities
tqdm                      # Progress bars
python-dotenv             # .env file loading
rapidfuzz                 # Fuzzy string matching
tiktoken                  # Token counting (OpenAI tokenizer)
```

**Critical install**: llama-cpp-python must be compiled with Metal support:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir
```

**Remaining deps**:
```bash
pip install litellm openai anthropic trafilatura beautifulsoup4 lxml markdownify html5lib \
  sentence-transformers scikit-learn pandas numpy matplotlib seaborn jsonschema \
  datasets huggingface-hub tqdm python-dotenv rapidfuzz tiktoken
```

### 2.3 Create Project Directory Structure

```bash
mkdir -p shared
mkdir -p idea-b-schema-pruning/{src,data/{raw_html,annotations,processed},results,notebooks,figures}
mkdir -p idea-c-quantization/{src,data,results,notebooks,figures}
mkdir -p models/{qwen2.5-3b,llama-3.2-3b,phi-3.5-mini,mistral-7b}
mkdir -p paper-b/figures
mkdir -p paper-c/figures
mkdir -p docs
```

Create `__init__.py` files:
```bash
touch shared/__init__.py
touch idea-b-schema-pruning/__init__.py
touch idea-b-schema-pruning/src/__init__.py
touch idea-c-quantization/__init__.py
touch idea-c-quantization/src/__init__.py
```

### 2.4 Set Up API Keys

Create `.env` file (NOT committed to git):
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Verify** keys exist:
```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'Missing OPENAI_API_KEY'
assert os.getenv('ANTHROPIC_API_KEY'), 'Missing ANTHROPIC_API_KEY'
print('API keys configured')
"
```

### 2.5 Install and Verify Gemini CLI

```bash
# Check if gemini CLI is installed
which gemini || echo "Need to install gemini CLI"

# Install via npm if needed
npm install -g @anthropic-ai/gemini-cli  # or appropriate package

# Test
echo "Hello, respond with just OK" | gemini
```

**Fallback**: If Gemini CLI unavailable or unreliable, use Google AI Studio API via litellm:
```python
import litellm
response = litellm.completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Say OK"}],
)
```

### 2.6 Download First Model

Download Qwen2.5-3B-Instruct Q4_K_M GGUF:
```bash
source .venv/bin/activate
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
  qwen2.5-3b-instruct-q4_k_m.gguf \
  --local-dir models/qwen2.5-3b
```

**Expected**: ~2.0 GB download, file at `models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf`

### 2.7 Verify Metal GPU Inference

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,      # offload all layers to Metal GPU
    n_ctx=4096,            # context window
    verbose=True,          # show Metal acceleration info
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
        {"role": "user", "content": 'Extract: {"name": "test product", "price": 29.99}. Output JSON.'},
    ],
    max_tokens=200,
    temperature=0,
)

print(output["choices"][0]["message"]["content"])
```

**Expected output**: Valid JSON with extracted data. Verbose log should show "Metal" or "GPU" acceleration lines.

### 2.8 Verify API Connections

```python
# OpenAI
from openai import OpenAI
client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=5,
)
print(f"OpenAI: {r.choices[0].message.content}")

# Anthropic
import anthropic
client = anthropic.Anthropic()
r = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=5,
    messages=[{"role": "user", "content": "Say OK"}],
)
print(f"Anthropic: {r.content[0].text}")
```

---

## 3. Acceptance Criteria

- [ ] `python -c "from llama_cpp import Llama; print('OK')"` succeeds
- [ ] `python -c "import litellm; print('OK')"` succeeds
- [ ] `python -c "import trafilatura; print('OK')"` succeeds
- [ ] `python -c "from sentence_transformers import SentenceTransformer; print('OK')"` succeeds
- [ ] Qwen2.5-3B Q4_K_M model file exists at `models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf`
- [ ] Model loads and generates valid JSON with Metal acceleration (n_gpu_layers=-1)
- [ ] OpenAI API call succeeds (1 test call)
- [ ] Anthropic API call succeeds (1 test call)
- [ ] Gemini CLI or API responds to test prompt
- [ ] All directories created per project structure
- [ ] `.env` file with API keys exists and loads correctly

---

## 4. Verification Script

```bash
#!/bin/bash
# verify_setup.sh
source .venv/bin/activate

echo "=== Checking Python ==="
python --version

echo "=== Checking imports ==="
python -c "from llama_cpp import Llama; print('llama_cpp: OK')"
python -c "import litellm; print('litellm: OK')"
python -c "import trafilatura; print('trafilatura: OK')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence_transformers: OK')"
python -c "from bs4 import BeautifulSoup; print('beautifulsoup4: OK')"
python -c "import jsonschema; print('jsonschema: OK')"
python -c "import pandas; print('pandas: OK')"
python -c "import matplotlib; print('matplotlib: OK')"
python -c "import tiktoken; print('tiktoken: OK')"
python -c "from rapidfuzz import fuzz; print('rapidfuzz: OK')"

echo "=== Checking model file ==="
ls -lh models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf

echo "=== Checking API keys ==="
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
for k in keys:
    v = os.getenv(k)
    print(f'{k}: {\"SET\" if v else \"MISSING\"} ({len(v) if v else 0} chars)')
"

echo "=== Checking directories ==="
for d in shared idea-b-schema-pruning/{src,data/{raw_html,annotations,processed},results,notebooks,figures} \
         idea-c-quantization/{src,data,results,notebooks,figures} \
         models/{qwen2.5-3b,llama-3.2-3b,phi-3.5-mini,mistral-7b} \
         paper-b/figures paper-c/figures docs; do
    [ -d "$d" ] && echo "  $d: OK" || echo "  $d: MISSING"
done

echo "=== All checks complete ==="
```

---

## 5. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| llama-cpp-python Metal compile failure | Low | High | Use pre-built wheel: `pip install llama-cpp-python` (may auto-detect Metal). Fallback: mlx-lm for Apple Silicon native inference |
| Python 3.14 compatibility issues | Medium | Medium | Some packages may not have 3.14 wheels yet. Fallback: use pyenv to install 3.12 |
| Model download slow/interrupted | Medium | Low | Start download early in background. HuggingFace CDN is fast (~50 MB/s). Can resume interrupted downloads |
| API keys missing/invalid | Low | High | User must provide keys. Test with minimal API call before proceeding |
| Gemini CLI not available | Medium | Low | Fallback to Google AI Studio API via litellm or 2-model consensus |
| Disk space insufficient for models | Low | Medium | All models for Phase 1 need ~2 GB. Full project needs ~60 GB. Check `df -h` |

---

## 6. Output Files

After this phase, the following should exist:
```
iitsrc/
├── .venv/                          # Python virtual environment (active)
├── .env                            # API keys (gitignored)
├── requirements.txt                # All dependencies listed
├── shared/__init__.py
├── idea-b-schema-pruning/src/
├── idea-b-schema-pruning/data/{raw_html,annotations,processed}/
├── idea-b-schema-pruning/{results,notebooks,figures}/
├── idea-c-quantization/src/
├── idea-c-quantization/{data,results,notebooks,figures}/
├── models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf   # ~2 GB
├── models/{llama-3.2-3b,phi-3.5-mini,mistral-7b}/       # empty, for later
├── paper-b/figures/
├── paper-c/figures/
└── docs/                           # Specification + phase docs
```

---

## 7. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Create venv + install deps | 15 min |
| llama-cpp-python Metal compilation | 5-10 min |
| Create directory structure | 5 min |
| Set up API keys + verify | 10 min |
| Download Qwen2.5-3B Q4_K_M (~2 GB) | 5-10 min |
| Verify Metal inference | 10 min |
| Verify API connections | 10 min |
| Gemini CLI setup | 10 min |
| Run verification script | 5 min |
| Buffer for troubleshooting | 20 min |
| **Total** | **~90-120 min** |
