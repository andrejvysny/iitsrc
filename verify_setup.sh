#!/bin/bash
# Verify complete project setup for IIT.SRC 2026

cd "$(dirname "$0")"

PASS=0
FAIL=0
WARN=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  [OK] $1"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $1"
        FAIL=$((FAIL + 1))
    fi
}

warn() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  [OK] $1"
        PASS=$((PASS + 1))
    else
        echo "  [WARN] $1 (optional)"
        WARN=$((WARN + 1))
    fi
}

echo "=== Directory Structure ==="
check "shared/" "[ -d shared ]"
check "idea-b-schema-pruning/src/" "[ -d idea-b-schema-pruning/src ]"
check "idea-c-quantization/src/" "[ -d idea-c-quantization/src ]"
check "models/" "[ -d models ]"
check "paper-b/" "[ -d paper-b ]"
check "paper-c/" "[ -d paper-c ]"

echo ""
echo "=== Core Modules ==="
check "shared/schemas.py" "[ -f shared/schemas.py ]"
check "shared/metrics.py" "[ -f shared/metrics.py ]"
check "shared/prompts.py" "[ -f shared/prompts.py ]"
check "shared/utils.py" "[ -f shared/utils.py ]"
check "shared/__init__.py" "[ -f shared/__init__.py ]"

echo ""
echo "=== Python Imports ==="
check "llama_cpp" "python -c 'import llama_cpp'"
check "litellm" "python -c 'import litellm'"
check "trafilatura" "python -c 'import trafilatura'"
check "sentence_transformers" "python -c 'import sentence_transformers'"
check "bs4" "python -c 'import bs4'"
check "jsonschema" "python -c 'import jsonschema'"
check "pandas" "python -c 'import pandas'"
check "matplotlib" "python -c 'import matplotlib'"
check "tiktoken" "python -c 'import tiktoken'"
check "rapidfuzz" "python -c 'import rapidfuzz'"
check "markdownify" "python -c 'import markdownify'"
check "sklearn" "python -c 'import sklearn'"
check "numpy" "python -c 'import numpy'"
check "openai" "python -c 'import openai'"
check "anthropic" "python -c 'import anthropic'"

echo ""
echo "=== Shared Module Imports ==="
check "schemas.get_schema" "python -c 'from shared.schemas import get_schema; get_schema(\"ecommerce\")'"
check "metrics.field_f1" "python -c 'from shared.metrics import field_f1'"
check "prompts.build_extraction_prompt" "python -c 'from shared.prompts import build_extraction_prompt'"
check "utils.parse_json_safe" "python -c 'from shared.utils import parse_json_safe, timer, count_tokens'"

echo ""
echo "=== Model Files ==="
warn "Qwen2.5-3B Q4_K_M" "[ -f models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf ]"

echo ""
echo "=== API Keys ==="
warn "OPENAI_API_KEY" "[ -n \"\$OPENAI_API_KEY\" ]"
warn "ANTHROPIC_API_KEY" "[ -n \"\$ANTHROPIC_API_KEY\" ]"
warn "OPENROUTER_API_KEY" "[ -n \"\$OPENROUTER_API_KEY\" ]"

echo ""
echo "=== Metal GPU ==="
warn "Metal backend" "python -c 'from llama_cpp import Llama; print(\"OK\")'"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed, $WARN warnings"
if [ $FAIL -gt 0 ]; then
    echo "STATUS: INCOMPLETE"
    exit 1
else
    echo "STATUS: READY"
fi
