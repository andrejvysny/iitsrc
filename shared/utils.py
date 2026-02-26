"""Shared utilities for extraction experiments."""

import json
import re
import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager for measuring execution time in seconds.

    Usage:
        with timer() as t:
            do_something()
        print(t["elapsed"])
    """
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def parse_json_safe(text: str) -> dict | None:
    """Extract and parse JSON from LLM output, handling common issues.

    Tries: direct parse, code block extraction, brace extraction.
    Returns parsed dict or None on failure.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken. Falls back to word estimate."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text.split()) * 4 // 3  # rough estimate
