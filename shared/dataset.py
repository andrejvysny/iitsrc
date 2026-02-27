"""ScrapeGraphAI-100k dataset loader, filter, sampler, and cacher."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get(
    "IITSRC_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "scrapegraphai"),
))
SAMPLE_CACHE = DATA_DIR / "sample_500.json"


def load_scrapegraphai() -> pd.DataFrame:
    """Load ScrapeGraphAI-100k from HuggingFace via datasets library."""
    from datasets import load_dataset
    ds = load_dataset("ScrapeGraphAI/scrapegraphai-100k", split="train")
    return ds.to_pandas()


def parse_schema(text: str) -> dict | None:
    """Parse schema string to dict. Returns None on failure."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    import re
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try brace extraction
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_response(text: str) -> dict | None:
    """Parse LLM response string to dict. Same logic as parse_schema."""
    return parse_schema(text)


def _count_schema_properties(schema: dict) -> int:
    """Count total properties including nested."""
    count = 0
    props = schema.get("properties", {})
    count += len(props)
    for v in props.values():
        if isinstance(v, dict):
            if v.get("type") == "object":
                count += _count_schema_properties(v)
            elif v.get("type") == "array" and isinstance(v.get("items"), dict):
                if v["items"].get("type") == "object":
                    count += _count_schema_properties(v["items"])
    return count


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid, appropriately-sized records."""
    mask = pd.Series(True, index=df.index)

    # Valid response
    if "response_is_valid" in df.columns:
        mask &= df["response_is_valid"] == True  # noqa: E712

    # Schema depth 2-6
    if "schema_depth" in df.columns:
        mask &= df["schema_depth"].between(2, 6)

    # Schema keys 3-50
    if "schema_keys" in df.columns:
        mask &= df["schema_keys"].between(3, 50)

    # Content length 200-100k chars
    if "content" in df.columns:
        content_len = df["content"].str.len()
        mask &= content_len.between(200, 100_000)

    # Source URL starts with http
    if "source" in df.columns:
        mask &= df["source"].str.startswith("http", na=False)

    return df[mask].reset_index(drop=True)


def sample_pages(df: pd.DataFrame, n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Stratified sample by schema_complexity_score quartiles."""
    if len(df) <= n:
        return df.copy()

    if "schema_complexity_score" in df.columns:
        df = df.copy()
        df["_quartile"] = pd.qcut(
            df["schema_complexity_score"], q=4, labels=False, duplicates="drop"
        )
        samples = []
        per_bin = n // df["_quartile"].nunique()
        remainder = n - per_bin * df["_quartile"].nunique()

        for i, (_, group) in enumerate(df.groupby("_quartile")):
            take = per_bin + (1 if i < remainder else 0)
            take = min(take, len(group))
            samples.append(group.sample(n=take, random_state=seed))

        result = pd.concat(samples).drop(columns=["_quartile"])
        # If we still need more due to small bins
        if len(result) < n:
            remaining = df.drop(result.index)
            extra = remaining.sample(n=min(n - len(result), len(remaining)), random_state=seed)
            result = pd.concat([result, extra])
        return result.reset_index(drop=True)
    else:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _build_record(row: pd.Series) -> dict[str, Any] | None:
    """Build a normalized record from a DataFrame row."""
    schema = parse_schema(str(row.get("schema", "")))
    gold = parse_response(str(row.get("response", "")))
    if schema is None or gold is None:
        return None

    row_id = str(row.get("id", row.name))
    return {
        "id": row_id,
        "source_url": str(row.get("source", "")),
        "prompt": str(row.get("prompt", "")),
        "schema": schema,
        "content_markdown": str(row.get("content", "")),
        "gold": gold,
        "schema_depth": int(row.get("schema_depth", 0)),
        "schema_keys": int(row.get("schema_keys", 0)),
        "schema_complexity_score": float(row.get("schema_complexity_score", 0.0)),
    }


def load_and_cache_sample(n: int = 500, force_reload: bool = False) -> list[dict]:
    """Load, filter, sample, and cache n records. Returns list of record dicts."""
    if SAMPLE_CACHE.exists() and not force_reload:
        with open(SAMPLE_CACHE) as f:
            records = json.load(f)
        if len(records) >= n:
            return records[:n]

    print(f"Loading ScrapeGraphAI-100k from HuggingFace...")
    df = load_scrapegraphai()
    print(f"  Loaded {len(df)} rows")

    df = filter_candidates(df)
    print(f"  After filtering: {len(df)} candidates")

    df = sample_pages(df, n=n)
    print(f"  Sampled {len(df)} pages")

    records = []
    for _, row in df.iterrows():
        rec = _build_record(row)
        if rec is not None:
            records.append(rec)
    print(f"  Built {len(records)} valid records")

    # Cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_CACHE, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Cached to {SAMPLE_CACHE}")

    return records
