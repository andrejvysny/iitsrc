"""Inference engine for quantization study — all model×quant variants."""

import gc
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from shared.prompts import build_extraction_prompt
from shared.utils import parse_json_safe, timer

# Suppress llama.cpp Python-level logging
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
logging.getLogger("llama_cpp").setLevel(logging.ERROR)


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

MODELS_DIR = Path(os.environ.get(
    "IITSRC_MODELS_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "models"),
))

# Model family → (directory, filename prefix, chat_format)
MODEL_FAMILIES = {
    "qwen2.5-3b": ("qwen2.5-3b", "qwen2.5-3b-instruct", "chatml"),
    "llama-3.2-3b": ("llama-3.2-3b", "llama-3.2-3b-instruct", "llama-3"),
    "phi-3.5-mini": ("phi-3.5-mini", "phi-3.5-mini-instruct", "phi-3"),
    "mistral-7b": ("mistral-7b", "mistral-7b-instruct-v0.3", "mistral-instruct"),
}

# Available quants per model (verified against disk)
MODEL_QUANTS = {
    "qwen2.5-3b": ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
    "llama-3.2-3b": ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_l"],
    "phi-3.5-mini": ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
    "mistral-7b": ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"],
}

CHAT_FORMATS = {
    "qwen2.5-3b": "chatml",
    "llama-3.2-3b": "llama-3",
    "phi-3.5-mini": "phi-3",
    "mistral-7b": "mistral-instruct",
}


def get_model_path(model: str, quant: str) -> Path:
    """Get GGUF file path for a model×quant combination."""
    dirname, prefix, _ = MODEL_FAMILIES[model]
    return MODELS_DIR / dirname / f"{prefix}-{quant}.gguf"


def get_model_size_gb(model: str, quant: str) -> float:
    """Get model file size in GB."""
    path = get_model_path(model, quant)
    if path.exists():
        return path.stat().st_size / (1024 ** 3)
    return 0.0


def get_all_variants(check_disk: bool = True) -> list[tuple[str, str]]:
    """Get all (model, quant) combinations. Set check_disk=False to list all without verifying files."""
    variants = []
    for model, quants in MODEL_QUANTS.items():
        for quant in quants:
            if check_disk:
                path = get_model_path(model, quant)
                if not path.exists():
                    continue
            variants.append((model, quant))
    return variants


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


# Single-variant cache: keep one model loaded across pages
_cached_key: tuple[str, str] | None = None
_cached_llm = None


def _get_llm(model: str, quant: str, n_ctx: int):
    """Get or load model. Caches one variant; unloads previous if different."""
    global _cached_key, _cached_llm

    key = (model, quant)
    if _cached_key == key and _cached_llm is not None:
        return _cached_llm

    # Unload previous
    if _cached_llm is not None:
        del _cached_llm
        _cached_llm = None
        _cached_key = None
        gc.collect()

    from llama_cpp import Llama

    model_path = get_model_path(model, quant)
    chat_format = CHAT_FORMATS[model]

    with _suppress_stderr():
        _cached_llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            chat_format=chat_format,
            verbose=False,
        )
    _cached_key = key
    return _cached_llm


def unload_model() -> None:
    """Explicitly unload cached model to free memory."""
    global _cached_key, _cached_llm
    if _cached_llm is not None:
        del _cached_llm
        _cached_llm = None
        _cached_key = None
        gc.collect()


def load_and_extract(
    model: str,
    quant: str,
    content: str,
    schema: dict,
) -> dict[str, Any]:
    """Run extraction, reusing cached model when variant unchanged."""
    if model not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model: {model}")
    if quant not in MODEL_QUANTS.get(model, []):
        raise ValueError(f"Quant {quant} not available for {model}")

    n_ctx = _get_n_ctx(schema)
    llm = _get_llm(model, quant, n_ctx)

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
    model_size = get_model_size_gb(model, quant)

    return {
        "raw_output": raw,
        "parsed": parse_json_safe(raw),
        "latency_s": t["elapsed"],
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "model": model,
        "quant": quant,
        "model_size_gb": round(model_size, 2),
    }
