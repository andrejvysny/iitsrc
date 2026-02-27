"""JSONL manifest for tracking downloaded pages with resume support."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILE = "manifest.jsonl"


class Manifest:
    """Append-only JSONL log for crawl tracking and resume."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / MANIFEST_FILE
        self._downloaded: set[str] = set()
        self._content_hashes: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing manifest entries for resume."""
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "ok":
                self._downloaded.add(entry["filename"])
                if h := entry.get("content_hash"):
                    self._content_hashes.add(h)

    def is_downloaded(self, filename: str) -> bool:
        return filename in self._downloaded

    def is_duplicate_content(self, html: str) -> bool:
        """Check near-duplicate via content hash (title + first 2000 chars)."""
        h = self._content_hash(html)
        return h in self._content_hashes

    def append(
        self,
        *,
        url: str,
        filename: str,
        site: str,
        category: str,
        status: str,
        content_hash: str = "",
        size_bytes: int = 0,
        error: str = "",
    ) -> None:
        """Append entry to manifest."""
        entry: dict[str, Any] = {
            "url": url,
            "filename": filename,
            "site": site,
            "category": category,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if content_hash:
            entry["content_hash"] = content_hash
        if size_bytes:
            entry["size_bytes"] = size_bytes
        if error:
            entry["error"] = error

        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        if status == "ok":
            self._downloaded.add(filename)
            if content_hash:
                self._content_hashes.add(content_hash)

    def record_ok(
        self, *, url: str, filename: str, site: str, category: str, html: str
    ) -> None:
        self.append(
            url=url,
            filename=filename,
            site=site,
            category=category,
            status="ok",
            content_hash=self._content_hash(html),
            size_bytes=len(html.encode()),
        )

    def record_fail(
        self, *, url: str, filename: str, site: str, category: str, error: str
    ) -> None:
        self.append(
            url=url,
            filename=filename,
            site=site,
            category=category,
            status="failed",
            error=error,
        )

    @staticmethod
    def _content_hash(html: str) -> str:
        """Hash title + middle section for dedup. First 2000 chars are
        often identical boilerplate on Amazon/eBay, so we sample from
        the middle of the page where product-specific content lives."""
        import re
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        mid = len(html) // 2
        sample = html[mid : mid + 3000]
        return hashlib.md5(f"{title}|{sample}".encode()).hexdigest()

    def stats(self) -> dict[str, int]:
        """Count entries by status."""
        counts: dict[str, int] = {}
        if not self.path.exists():
            return counts
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                s = entry.get("status", "unknown")
                counts[s] = counts.get(s, 0) + 1
            except json.JSONDecodeError:
                continue
        return counts
