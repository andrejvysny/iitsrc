"""Create stratified val/test split from crawled dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "idea-b-schema-pruning" / "data"
RAW_DIR = DATA_DIR / "raw_html"
SPLIT_PATH = DATA_DIR / "split.json"

SEED = 42
VAL_FRACTION = 0.10  # 10% validation


def collect_files() -> dict[str, list[str]]:
    """Group files by site × category for stratification."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in sorted(RAW_DIR.glob("*.html")):
        parts = f.stem.split("_")
        if len(parts) >= 4:
            domain = parts[0]
            site = parts[1]
            category = "_".join(parts[2:-1])
            key = f"{domain}_{site}_{category}"
            groups[key].append(f.name)
    return groups


def stratified_split(
    groups: dict[str, list[str]],
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    """Split each group proportionally into val/test."""
    rng = random.Random(seed)
    val_files: list[str] = []
    test_files: list[str] = []

    for key in sorted(groups):
        files = groups[key][:]
        rng.shuffle(files)
        n_val = max(1, round(len(files) * val_fraction))
        val_files.extend(files[:n_val])
        test_files.extend(files[n_val:])

    return val_files, test_files


def main() -> None:
    groups = collect_files()
    total = sum(len(v) for v in groups.values())

    if total == 0:
        print("No HTML files found. Run crawlers first.")
        return

    val, test = stratified_split(groups)

    split = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "total": total,
        "val_count": len(val),
        "test_count": len(test),
        "val": sorted(val),
        "test": sorted(test),
    }

    SPLIT_PATH.write_text(json.dumps(split, indent=2))

    print(f"Total files: {total}")
    print(f"Validation:  {len(val)} ({len(val)/total*100:.1f}%)")
    print(f"Test:        {len(test)} ({len(test)/total*100:.1f}%)")
    print(f"Groups:      {len(groups)} (site × category)")
    print(f"Written to:  {SPLIT_PATH}")


if __name__ == "__main__":
    main()
