"""HTML quality validation for crawled pages."""

from __future__ import annotations

import re
from dataclasses import dataclass

# CAPTCHA / block indicators — only checked on small pages (<50KB)
# to avoid false positives from CSS classes like .grecaptcha-badge
CAPTCHA_PATTERNS: list[str] = [
    "robot check",
    "are you a human",
    "verify you are human",
    "automated access to amazon",
    "sorry, we just need to make sure you're not a robot",
    "please verify you are a human",
    "unusual traffic from your computer",
    "press & hold to confirm you are a human",
]

# These are checked separately — presence on a small page = definite block
BLOCK_PATTERNS: list[str] = [
    "access denied",
    "perimeterx",
    "distil networks",
    "datadome",
]

# Error page indicators
ERROR_PATTERNS: list[str] = [
    "page not found",
    "404 error",
    "this page isn't available",
    "item not found",
    "listing has been removed",
    "no longer available",
    "sorry, this listing",
    "this listing has ended",
]

# Content markers per domain
ECOM_CONTENT_MARKERS: list[str] = [
    "price",
    "add to cart",
    "buy now",
    "add to basket",
    "in stock",
    "shipping",
    "product",
]

RE_CONTENT_MARKERS: list[str] = [
    "bed",
    "bath",
    "sqft",
    "sq ft",
    "square",
    "price",
    "for sale",
    "listing",
    "property",
    "m²",
    "izby",  # Slovak: rooms
    "cena",  # Slovak: price
    "plocha",  # Slovak: area
]


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


def validate_html(
    html: str,
    *,
    domain: str = "ecommerce",
    min_size: int = 5_000,
) -> ValidationResult:
    """Run quality checks on downloaded HTML."""
    if len(html.encode()) < min_size:
        return ValidationResult(False, f"too_small ({len(html.encode())} bytes)")

    html_lower = html.lower()

    if _has_captcha(html_lower):
        return ValidationResult(False, "captcha_detected")

    if _is_error_page(html_lower):
        return ValidationResult(False, "error_page")

    markers = ECOM_CONTENT_MARKERS if domain == "ecommerce" else RE_CONTENT_MARKERS
    if not _has_content_markers(html_lower, markers, threshold=2):
        return ValidationResult(False, "no_content_markers")

    return ValidationResult(True)


def _has_captcha(html_lower: str) -> bool:
    # Large pages with captcha strings are normal (CSS class names, etc.)
    if len(html_lower) > 50_000:
        return False
    if any(p in html_lower for p in CAPTCHA_PATTERNS):
        return True
    if any(p in html_lower for p in BLOCK_PATTERNS):
        return True
    return False


def _is_error_page(html_lower: str) -> bool:
    # Must match at least 1 error pattern AND be a short page
    if len(html_lower) > 50_000:
        return False  # large pages are unlikely pure error pages
    return any(p in html_lower for p in ERROR_PATTERNS)


def _has_content_markers(
    html_lower: str, markers: list[str], threshold: int = 2
) -> bool:
    """Check that at least `threshold` content markers appear."""
    found = sum(1 for m in markers if m in html_lower)
    return found >= threshold


def extract_title(html: str) -> str:
    """Extract <title> text for logging."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip()[:120] if m else ""
