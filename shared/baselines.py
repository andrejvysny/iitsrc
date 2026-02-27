"""Cloud API baselines via litellm."""

import json
from typing import Any

import litellm
from dotenv import load_dotenv

from shared.prompts import build_messages
from shared.utils import parse_json_safe, timer

load_dotenv()

CLOUD_MODELS = {
    "gpt-4o": "gpt-4o",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "qwen-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
    "llama-70b": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "mistral-large": "openrouter/mistralai/mistral-large",
}


def run_cloud_model(
    content: str,
    schema: dict,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Run extraction via cloud model. Returns result dict."""
    model_id = CLOUD_MODELS.get(model, model)
    messages = build_messages(schema, content)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # JSON mode for OpenAI models
    if model_id.startswith("gpt-"):
        kwargs["response_format"] = {"type": "json_object"}

    with timer() as t:
        response = litellm.completion(**kwargs)

    raw = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "raw_output": raw,
        "parsed": parse_json_safe(raw),
        "latency_s": t["elapsed"],
        "model": model,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }
