"""HTML cleaning for annotation and extraction pipelines."""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup, Comment

from shared.utils import count_tokens

REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "iframe", "noscript", "svg", "link", "meta", "picture",
    "source", "video", "audio", "canvas", "map",
}

STRIP_ATTRS = {"style", "class", "onclick", "onload", "onmouseover",
               "onmouseout", "onfocus", "onblur", "onchange", "onsubmit"}

# id/class patterns for cookie banners, consent popups, ads, recommendations
BOILERPLATE_RE = re.compile(
    r"cookie|consent|gdpr|privacy-banner|ad-container|advertisement|"
    r"popup-overlay|newsletter-signup|onetrust|didomi|sp_message|"
    r"recommendation|similar.?item|also.?view|also.?like|sponsored|"
    r"carousel|slider|trending|recently.?view|related.?search",
    re.IGNORECASE,
)


def _should_remove_element(tag) -> bool:
    """Check if element is ad/consent/recommendation container by id or class."""
    for attr in ("id", "class"):
        val = tag.get(attr)
        if val:
            text = val if isinstance(val, str) else " ".join(val)
            if BOILERPLATE_RE.search(text):
                return True
    return False


def _extract_json_ld(soup: BeautifulSoup) -> tuple[dict, str]:
    """Extract JSON-LD structured data before script removal.

    Returns (raw_dict, text_prefix). Excludes price/currency — seller's
    displayed price is ground truth, not JSON-LD's potentially converted value.
    JSON-LD is authoritative for: rating, availability, itemCondition.
    """
    raw_data: dict = {}
    lines: list[str] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Normalize: handle top-level array or @graph wrapper
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "@graph" in data:
            items = data["@graph"]
        elif isinstance(data, dict):
            items = [data]
        else:
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_data.update(item)

            # Safe non-price fields only
            for key in ("name", "itemCondition", "availability"):
                if key in item:
                    lines.append(f"{key}: {item[key]}")

            agg = item.get("aggregateRating")
            if isinstance(agg, dict):
                for sub in ("ratingValue", "reviewCount"):
                    if sub in agg:
                        lines.append(f"aggregateRating.{sub}: {agg[sub]}")

    text_block = ""
    if lines:
        text_block = "## Structured Product Data\n" + "\n".join(lines) + "\n\n"

    return raw_data, text_block


def clean_html(html: str, max_tokens: int = 30_000) -> tuple[str, dict]:
    """Clean HTML by removing non-content elements and attributes.

    Extracts JSON-LD structured data (non-price fields) before script removal
    and prepends it as a plain-text block for model grounding.

    Returns (cleaned_content, json_ld_dict).
    Falls back to plain text extraction if cleaned HTML exceeds max_tokens.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract JSON-LD before script tags are removed
    json_ld, json_ld_prefix = _extract_json_ld(soup)

    # Remove unwanted tags
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Remove boilerplate containers (ads, consent, recommendations)
    for tag in list(soup.find_all(True)):
        if tag.parent and _should_remove_element(tag):
            tag.decompose()

    # Remove empty tags (except self-closing like br, hr, img)
    for tag in list(soup.find_all(True)):
        if tag.parent is None:
            continue
        if tag.name not in {"br", "hr", "img", "input"} and not tag.get_text(strip=True) and not tag.find_all("img"):
            tag.decompose()

    # Strip unwanted attributes
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        attrs_to_remove = []
        for attr in list(tag.attrs):
            if attr in STRIP_ATTRS or attr.startswith("data-") or attr.startswith("on"):
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del tag[attr]

    cleaned = str(soup)

    # Token budget check — fallback to text-only extraction
    if count_tokens(cleaned) > max_tokens:
        text = soup.get_text(separator="\n", strip=True)
        # Deduplicate consecutive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        if count_tokens(text) > max_tokens:
            text = truncate_to_tokens(text, max_tokens)
        return json_ld_prefix + text, json_ld

    return json_ld_prefix + cleaned, json_ld


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens using binary search."""
    if count_tokens(text) <= max_tokens:
        return text
    # Approximate: 1 token ≈ 4 chars, then refine
    estimate = max_tokens * 4
    if estimate >= len(text):
        estimate = len(text)
    # Binary search for the right cutoff
    low, high = estimate // 2, min(estimate * 2, len(text))
    while low < high - 100:
        mid = (low + high) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid
    return text[:low]
