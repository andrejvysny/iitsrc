"""Local GGUF inference for Idea B (Q4_K_M, 3 models)."""

import gc
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Suppress llama.cpp Python-level logging
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
logging.getLogger("llama_cpp").setLevel(logging.ERROR)

from shared.prompts import build_extraction_prompt
from shared.utils import parse_json_safe, timer


@contextmanager
def _suppress_stderr():
    """Suppress C-level stderr output (ggml_metal_init bf16 warnings)."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

MODEL_CONFIGS = {
    "qwen2.5-3b": {
        "path": MODELS_DIR / "qwen2.5-3b" / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
    },
    "llama-3.2-3b": {
        "path": MODELS_DIR / "llama-3.2-3b" / "llama-3.2-3b-instruct-q4_k_m.gguf",
        "chat_format": "llama-3",
    },
    "phi-3.5-mini": {
        "path": MODELS_DIR / "phi-3.5-mini" / "phi-3.5-mini-instruct-q4_k_m.gguf",
        "chat_format": "phi-3",
    },
}

# Single-model cache
_current_model_name: str | None = None
_current_llm = None


def _count_properties(schema: dict) -> int:
    """Count top-level + nested properties for n_ctx sizing."""
    count = 0
    for v in schema.get("properties", {}).values():
        count += 1
        if isinstance(v, dict):
            if v.get("type") == "object":
                count += len(v.get("properties", {}))
            elif v.get("type") == "array" and isinstance(v.get("items"), dict):
                count += len(v.get("items", {}).get("properties", {}))
    return count


def _get_n_ctx(schema: dict) -> int:
    """Dynamic n_ctx: 4096 for <10 properties, 8192 for 10+."""
    return 8192 if _count_properties(schema) >= 10 else 4096


def _load_model(model_name: str, n_ctx: int) -> Any:
    """Load model, unloading previous if different."""
    global _current_model_name, _current_llm

    if _current_model_name == model_name and _current_llm is not None:
        return _current_llm

    # Unload previous
    if _current_llm is not None:
        del _current_llm
        _current_llm = None
        _current_model_name = None
        gc.collect()

    from llama_cpp import Llama

    cfg = MODEL_CONFIGS[model_name]
    with _suppress_stderr():
        _current_llm = Llama(
            model_path=str(cfg["path"]),
            n_ctx=n_ctx,
            n_gpu_layers=-1,  # Metal: offload all
            chat_format=cfg["chat_format"],
            verbose=False,
        )
    _current_model_name = model_name
    return _current_llm


def unload_model() -> None:
    """Explicitly unload current model to free memory."""
    global _current_model_name, _current_llm
    if _current_llm is not None:
        del _current_llm
        _current_llm = None
        _current_model_name = None
        gc.collect()


def extract(content: str, schema: dict, model_name: str) -> dict[str, Any]:
    """Run extraction on content using a local GGUF model.

    Returns dict with raw_output, parsed, latency_s, tokens_in, tokens_out, model.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    n_ctx = _get_n_ctx(schema)
    llm = _load_model(model_name, n_ctx)

    prompt = build_extraction_prompt(schema, content)

    with timer() as t:
        result = llm(
            prompt,
            max_tokens=1024,
            temperature=0.0,
            stop=["```", "\n\n\n"],
        )

    raw = result["choices"][0]["text"] if result["choices"] else ""
    usage = result.get("usage", {})

    return {
        "raw_output": raw,
        "parsed": parse_json_safe(raw),
        "latency_s": t["elapsed"],
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "model": model_name,
    }
