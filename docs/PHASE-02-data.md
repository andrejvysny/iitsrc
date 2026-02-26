# PHASE 2: Data Collection

**Duration**: ~2 hours
**Schedule**: Day 1, 10:00-12:00
**Dependencies**: Phase 1 (environment ready)
**Blocks**: Phase 3 (annotation), Phase 5 (pruning experiments), Phase 6 (quantization experiments)

---

## 1. Objective

Collect 100 HTML pages (50 e-commerce + 50 real estate) from 8 websites across 2 languages (English + Slovak). Define JSON schemas in code. Download and sample ScrapeGraphAI-100k dataset.

---

## 2. Tasks

### 2.1 Define JSON Schemas in Code

**File**: `shared/schemas.py`

Already specified in SPECIFICATION.md. Must include:
- `ECOM_SCHEMA` — JSON Schema dict for e-commerce (8 fields)
- `REALESTATE_SCHEMA` — JSON Schema dict for real estate (8 fields)
- `ECOM_KEYWORDS` — Set of keywords for heuristic pruning
- `REALESTATE_KEYWORDS` — Set of keywords for heuristic pruning
- Helper functions: `get_schema(domain)`, `get_schema_field_names(domain)`, `get_schema_description(domain)`

### 2.2 Collect E-Commerce Pages (50 total)

#### Amazon.com (12-13 pages)

**Target**: Product detail pages (different categories)
**URL pattern**: `https://www.amazon.com/dp/{ASIN}` or `https://www.amazon.com/{product-name}/dp/{ASIN}`
**Collection method**: Open in browser → Cmd+S → Save as "Webpage, Complete" or "Webpage, HTML Only"
**Categories to cover**: Electronics, clothing, books, home goods, toys (diversity of layouts)

**Naming**: `ecom_amazon_001.html` through `ecom_amazon_013.html`

**What to verify after saving**:
- File is valid HTML (opens in browser)
- Product name visible in source
- Price visible in source
- Key product details present (not hidden behind JS that didn't render)

#### eBay.com (12-13 pages)

**Target**: Active listings (Buy It Now preferred, not auction)
**URL pattern**: `https://www.ebay.com/itm/{item-id}`
**Collection method**: Same as Amazon
**Naming**: `ecom_ebay_001.html` through `ecom_ebay_013.html`

#### Alza.sk (12-13 pages)

**Target**: Product detail pages
**URL pattern**: `https://www.alza.sk/{product-slug}-d{product-id}.htm`
**Language**: Slovak (some fields may be in Czech)
**Collection method**: Browser save (important: Slovak encoding UTF-8)
**Naming**: `ecom_alza_001.html` through `ecom_alza_013.html`

**Note**: Alza pages are typically well-structured with clear product info sections.

#### Mall.sk (12-13 pages)

**Target**: Product detail pages
**URL pattern**: `https://www.mall.sk/{product-slug}`
**Language**: Slovak
**Collection method**: Browser save
**Naming**: `ecom_mall_001.html` through `ecom_mall_013.html`

**Note**: If Mall.sk is unavailable or difficult, alternatives: Heureka.sk, CZC.cz, Datart.sk

### 2.3 Collect Real Estate Pages (50 total)

#### Zillow.com (12-13 pages)

**Target**: Property listing detail pages (for sale)
**URL pattern**: `https://www.zillow.com/homedetails/{address}/{zpid}_zpid/`
**Collection method**: Browser save
**Naming**: `realestate_zillow_001.html` through `realestate_zillow_013.html`
**Categories**: Mix of apartments, houses, condos

#### Realtor.com (12-13 pages)

**Target**: Property listing detail pages
**URL pattern**: `https://www.realtor.com/realestateandhomes-detail/{address}`
**Collection method**: Browser save
**Naming**: `realestate_realtor_001.html` through `realestate_realtor_013.html`

**Fallback**: If Realtor.com is difficult, use Redfin.com or Homes.com

#### Nehnutelnosti.sk (12-13 pages)

**Target**: Property listings (predaj/sale)
**URL pattern**: `https://www.nehnutelnosti.sk/detail/{id}/`
**Language**: Slovak
**Collection method**: Browser save
**Naming**: `realestate_nehnutelnosti_001.html` through `realestate_nehnutelnosti_013.html`

#### Reality.sk (12-13 pages)

**Target**: Property listings
**URL pattern**: `https://www.reality.sk/{listing-type}/{slug}/`
**Language**: Slovak
**Collection method**: Browser save
**Naming**: `realestate_reality_001.html` through `realestate_reality_013.html`

**Fallback**: If Reality.sk is unavailable, use Topreality.sk or Bezrealitky.sk

### 2.4 Page Collection Guidelines

**General rules**:
1. Save from browser (Cmd+S) to capture JS-rendered content
2. Save as "Webpage, HTML Only" (no images/CSS files needed)
3. Ensure UTF-8 encoding
4. Pick DIVERSE pages: different price ranges, categories, layouts
5. Avoid pages with unusual layouts (e.g., bulk listings, comparison pages)
6. Each page should have at least 5 of 8 schema fields extractable

**Diversity requirements**:
- At least 3 different product categories per e-commerce site
- Mix of properties with full info vs. partial info (some null fields expected)
- Include at least 2-3 "hard" pages per site (unusual layout, missing fields, non-standard formatting)

**Automated collection alternative** (if manual is too slow):
```python
import subprocess

urls = [
    ("ecom_amazon_001", "https://www.amazon.com/dp/B0EXAMPLE1"),
    ("ecom_amazon_002", "https://www.amazon.com/dp/B0EXAMPLE2"),
    # ...
]

for name, url in urls:
    subprocess.run([
        "curl", "-L",
        "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "-o", f"idea-b-schema-pruning/data/raw_html/{name}.html",
        url
    ])
```

**Warning**: curl/wget may miss JS-rendered content. Browser save is preferred for accuracy.

### 2.5 Download ScrapeGraphAI-100k Dataset

```python
from datasets import load_dataset

ds = load_dataset("Ilanit/ScrapeGraphAI_100k", split="train")
print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"First example keys: {list(ds[0].keys())}")
```

**Sampling strategy**:
- Sample 500 examples
- Stratify by schema complexity (count number of fields in extraction schema)
- Ensure diversity of source websites
- Save locally as JSON files or single JSONL

```python
import random
import json

# Examine dataset structure first
sample = ds[0]
print(json.dumps(sample, indent=2, default=str)[:2000])

# Sample 500 with diverse schemas
# (exact sampling logic depends on dataset schema — adjust after examining)
indices = random.sample(range(len(ds)), 500)
sampled = ds.select(indices)
sampled.save_to_disk("idea-b-schema-pruning/data/scrapegraphai_500")
```

### 2.6 Define Data Split

```python
import random

# Load all 100 page filenames
all_pages = sorted(Path("idea-b-schema-pruning/data/raw_html").glob("*.html"))
assert len(all_pages) == 100

# Stratified split: equal representation of each domain/site
random.seed(42)
# 20 validation (for threshold tuning), 80 test
# Ensure each site has ~4 val + ~9 test pages

val_pages = []
test_pages = []
for site in ["amazon", "ebay", "alza", "mall", "zillow", "realtor", "nehnutelnosti", "reality"]:
    site_pages = [p for p in all_pages if site in p.name]
    random.shuffle(site_pages)
    val_pages.extend(site_pages[:3])   # ~3 per site = ~24 val (round to 20)
    test_pages.extend(site_pages[3:])

# Save split
split = {
    "validation": [p.name for p in val_pages[:20]],
    "test": [p.name for p in test_pages + val_pages[20:]],
}
with open("idea-b-schema-pruning/data/split.json", "w") as f:
    json.dump(split, f, indent=2)
```

### 2.7 Symlink Data for Idea C

```bash
# Idea C uses same data as Idea B
ln -s ../idea-b-schema-pruning/data/raw_html idea-c-quantization/data/raw_html
ln -s ../idea-b-schema-pruning/data/annotations idea-c-quantization/data/annotations
ln -s ../idea-b-schema-pruning/data/split.json idea-c-quantization/data/split.json
```

---

## 3. Acceptance Criteria

- [ ] 50 e-commerce HTML files saved in `idea-b-schema-pruning/data/raw_html/`
  - [ ] 12-13 from Amazon.com
  - [ ] 12-13 from eBay.com
  - [ ] 12-13 from Alza.sk
  - [ ] 12-13 from Mall.sk
- [ ] 50 real estate HTML files saved in `idea-b-schema-pruning/data/raw_html/`
  - [ ] 12-13 from Zillow.com
  - [ ] 12-13 from Realtor.com
  - [ ] 12-13 from Nehnutelnosti.sk
  - [ ] 12-13 from Reality.sk
- [ ] Total: 100 HTML files
- [ ] Each file parseable by BeautifulSoup (`lxml` parser)
- [ ] Each file contains relevant product/property information in source HTML
- [ ] Naming convention followed: `{domain}_{site}_{NNN}.html`
- [ ] JSON schemas defined in `shared/schemas.py` with validation
- [ ] ScrapeGraphAI-100k: 500 examples sampled and saved locally
- [ ] Data split defined: `split.json` with ~20 validation + ~80 test pages
- [ ] Symlinks created for `idea-c-quantization/data/`

---

## 4. Verification Script

```python
# verify_data.py
from pathlib import Path
from bs4 import BeautifulSoup
import json

data_dir = Path("idea-b-schema-pruning/data/raw_html")
files = sorted(data_dir.glob("*.html"))
print(f"Total HTML files: {len(files)}")

# Count per site
sites = {}
for f in files:
    parts = f.stem.split("_")
    site = parts[1] if len(parts) >= 2 else "unknown"
    sites[site] = sites.get(site, 0) + 1

print("\nPer-site counts:")
for site, count in sorted(sites.items()):
    print(f"  {site}: {count}")

# Parse test
print("\nParsability test (first 5):")
for f in files[:5]:
    try:
        html = f.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        n_elements = len(soup.find_all())
        text_len = len(soup.get_text())
        print(f"  {f.name}: {n_elements} elements, {text_len} chars text")
    except Exception as e:
        print(f"  {f.name}: ERROR - {e}")

# Check split
split_path = Path("idea-b-schema-pruning/data/split.json")
if split_path.exists():
    split = json.loads(split_path.read_text())
    print(f"\nData split: {len(split['validation'])} val, {len(split['test'])} test")
else:
    print("\nWARNING: split.json not found")

# Check ScrapeGraphAI
sgai_dir = Path("idea-b-schema-pruning/data/scrapegraphai_500")
if sgai_dir.exists():
    print(f"\nScrapeGraphAI data: exists")
else:
    print(f"\nScrapeGraphAI data: not downloaded yet")
```

---

## 5. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Websites block automated access | High | Medium | Use browser Cmd+S for all pages. Manual but reliable |
| Slovak sites different encoding | Medium | Low | Force UTF-8 when reading: `f.read_text(encoding='utf-8', errors='replace')` |
| JS-rendered content missing | High | High | Must save from browser, not wget/curl. Chrome DevTools > Cmd+S captures rendered DOM |
| Some pages lack key fields | Medium | Low | Accept some null fields. Verify at least 5/8 fields extractable per page |
| Mall.sk or Reality.sk unavailable | Low | Low | Fallback sites listed (Heureka.sk, Topreality.sk) |
| ScrapeGraphAI-100k format unfamiliar | Medium | Low | Examine dataset structure first, adapt sampling |
| Too slow manual collection | Medium | Medium | Target 2-3 min per page. 100 pages × 2.5 min = ~4 hours — may need to reduce count or parallelize |

---

## 6. Output Files

```
idea-b-schema-pruning/data/
├── raw_html/
│   ├── ecom_amazon_001.html ... ecom_amazon_013.html
│   ├── ecom_ebay_001.html ... ecom_ebay_013.html
│   ├── ecom_alza_001.html ... ecom_alza_013.html
│   ├── ecom_mall_001.html ... ecom_mall_013.html
│   ├── realestate_zillow_001.html ... realestate_zillow_013.html
│   ├── realestate_realtor_001.html ... realestate_realtor_013.html
│   ├── realestate_nehnutelnosti_001.html ... realestate_nehnutelnosti_013.html
│   └── realestate_reality_001.html ... realestate_reality_013.html
├── split.json
└── scrapegraphai_500/              # HuggingFace datasets format

idea-c-quantization/data/
├── raw_html -> ../../idea-b-schema-pruning/data/raw_html
├── annotations -> ../../idea-b-schema-pruning/data/annotations
└── split.json -> ../../idea-b-schema-pruning/data/split.json
```

---

## 7. Time Breakdown

| Task | Estimated Time |
|------|---------------|
| Define schemas in code | 10 min (already partially done) |
| Collect 25 English e-commerce pages | 30 min |
| Collect 25 Slovak e-commerce pages | 30 min |
| Collect 25 English real estate pages | 30 min |
| Collect 25 Slovak real estate pages | 30 min |
| Download ScrapeGraphAI-100k + sample 500 | 15 min |
| Define split, create symlinks | 10 min |
| Verify all data | 10 min |
| **Total** | **~2.5 hours** |

**Note**: Page collection is the bottleneck. Can be parallelized by opening multiple browser tabs. If running behind, reduce to 10 pages per site (80 total) instead of 12-13.
