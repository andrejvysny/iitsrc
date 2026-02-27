"""Async HTML re-fetcher with file-based caching."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import aiohttp

HTML_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "scrapegraphai" / "html"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _cache_path(record_id: str) -> Path:
    """Get cache file path for a record ID."""
    safe_id = record_id.replace("/", "_").replace("\\", "_")
    if len(safe_id) > 200:
        safe_id = hashlib.md5(safe_id.encode()).hexdigest()
    return HTML_CACHE_DIR / f"{safe_id}.html"


def _load_cached(record_id: str) -> str | None:
    """Load cached HTML if it exists."""
    path = _cache_path(record_id)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def _save_cached(record_id: str, html: str) -> None:
    """Save HTML to cache."""
    HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(record_id)
    path.write_text(html, encoding="utf-8")


async def _fetch_one(
    session: aiohttp.ClientSession,
    record_id: str,
    url: str,
    semaphore: asyncio.Semaphore,
    delay: float = 0.2,
) -> tuple[str, str | None]:
    """Fetch a single URL. Returns (record_id, html_or_none)."""
    # Check cache first
    cached = _load_cached(record_id)
    if cached is not None:
        return record_id, cached

    async with semaphore:
        await asyncio.sleep(delay)
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    html = await resp.text(errors="replace")
                    if html and len(html) > 100:
                        _save_cached(record_id, html)
                        return record_id, html
                return record_id, None
        except Exception:
            return record_id, None


async def _fetch_all_async(
    records: list[dict[str, Any]],
    concurrency: int = 10,
    delay: float = 0.2,
) -> dict[str, str | None]:
    """Fetch HTML for all records asynchronously."""
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(ssl=False, limit=concurrency)

    results: dict[str, str | None] = {}
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, connector=connector) as session:
        tasks = [
            _fetch_one(session, rec["id"], rec["source_url"], semaphore, delay)
            for rec in records
            if rec.get("source_url", "").startswith("http")
        ]
        from tqdm.asyncio import tqdm_asyncio
        completed = await tqdm_asyncio.gather(*tasks, desc="Fetching HTML")
        for record_id, html in completed:
            results[record_id] = html

    return results


def fetch_all_html(
    records: list[dict[str, Any]],
    concurrency: int = 10,
    delay: float = 0.2,
) -> dict[str, str | None]:
    """Fetch HTML for all records. Returns {record_id: html_or_none}."""
    return asyncio.run(_fetch_all_async(records, concurrency, delay))


def get_fetch_stats(results: dict[str, str | None]) -> dict[str, Any]:
    """Compute fetch statistics."""
    total = len(results)
    success = sum(1 for v in results.values() if v is not None)
    return {
        "total": total,
        "success": success,
        "failed": total - success,
        "success_rate": success / total if total > 0 else 0.0,
    }
