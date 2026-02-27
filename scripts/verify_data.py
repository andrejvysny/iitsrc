"""Dataset integrity report: counts, parsability, captcha detection, dedup."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "idea-b-schema-pruning" / "data" / "raw_html"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"


def load_manifest() -> list[dict]:
    entries = []
    if not MANIFEST_PATH.exists():
        return entries
    for line in MANIFEST_PATH.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def count_files() -> dict[str, dict[str, int]]:
    """Count HTML files by site × category."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if not DATA_DIR.exists():
        return counts
    for f in DATA_DIR.glob("*.html"):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            domain = parts[0]
            site = parts[1]
            category = "_".join(parts[2:-1])  # handle multi-word categories
            counts[f"{domain}_{site}"][category] += 1
    return counts


def check_file_sizes() -> dict[str, list[str]]:
    """Find suspiciously small files."""
    issues: dict[str, list[str]] = {"small": [], "empty": []}
    if not DATA_DIR.exists():
        return issues
    for f in DATA_DIR.glob("*.html"):
        size = f.stat().st_size
        if size == 0:
            issues["empty"].append(f.name)
        elif size < 5000:
            issues["small"].append(f.name)
    return issues


CAPTCHA_PATTERNS = [
    "robot check", "are you a human", "verify you are human",
    "automated access to amazon", "sorry, we just need to make sure",
    "unusual traffic from your computer", "access denied",
    "perimeterx", "distil networks", "datadome",
]


def check_captcha_pages() -> list[str]:
    """Scan files for CAPTCHA indicators (only small files = likely block pages)."""
    flagged = []
    if not DATA_DIR.exists():
        return flagged
    for f in DATA_DIR.glob("*.html"):
        if f.stat().st_size > 50_000:
            continue  # large pages won't be captcha pages
        text = f.read_text(errors="ignore").lower()[:5000]
        if any(p in text for p in CAPTCHA_PATTERNS):
            flagged.append(f.name)
    return flagged


def check_duplicates() -> list[tuple[str, str]]:
    """Find near-duplicate files by content hash (title + mid-page sample)."""
    import hashlib
    import re

    hashes: dict[str, str] = {}
    dupes: list[tuple[str, str]] = []
    if not DATA_DIR.exists():
        return dupes
    for f in DATA_DIR.glob("*.html"):
        content = f.read_text(errors="ignore")
        title_m = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        title = title_m.group(1).strip() if title_m else ""
        mid = len(content) // 2
        sample = content[mid : mid + 3000]
        h = hashlib.md5(f"{title}|{sample}".encode()).hexdigest()
        if h in hashes:
            dupes.append((f.name, hashes[h]))
        else:
            hashes[h] = f.name
    return dupes


def main() -> None:
    print("=" * 60)
    print("DATASET INTEGRITY REPORT")
    print("=" * 60)

    # File counts
    counts = count_files()
    total = 0
    print("\n--- Files per site × category ---")
    for site_key in sorted(counts):
        cats = counts[site_key]
        site_total = sum(cats.values())
        total += site_total
        print(f"\n  {site_key} ({site_total} total):")
        for cat in sorted(cats):
            print(f"    {cat}: {cats[cat]}")
    print(f"\n  TOTAL FILES: {total}")

    # Manifest stats
    entries = load_manifest()
    if entries:
        print("\n--- Manifest stats ---")
        status_counts: dict[str, int] = defaultdict(int)
        for e in entries:
            status_counts[e.get("status", "unknown")] += 1
        for s, c in sorted(status_counts.items()):
            print(f"  {s}: {c}")

    # Size issues
    sizes = check_file_sizes()
    if sizes["empty"] or sizes["small"]:
        print("\n--- Size issues ---")
        if sizes["empty"]:
            print(f"  Empty files ({len(sizes['empty'])}): {sizes['empty'][:5]}")
        if sizes["small"]:
            print(f"  Small files <5KB ({len(sizes['small'])}): {sizes['small'][:5]}")
    else:
        print("\n--- Size issues: none ---")

    # CAPTCHA check
    captcha = check_captcha_pages()
    if captcha:
        print(f"\n--- CAPTCHA detected ({len(captcha)}) ---")
        for f in captcha[:10]:
            print(f"  {f}")
    else:
        print("\n--- CAPTCHA pages: none ---")

    # Duplicates
    dupes = check_duplicates()
    if dupes:
        print(f"\n--- Duplicates ({len(dupes)}) ---")
        for a, b in dupes[:10]:
            print(f"  {a} <-> {b}")
    else:
        print("\n--- Duplicates: none ---")

    print("\n" + "=" * 60)

    # Exit code: fail if major issues
    if captcha or sizes["empty"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
