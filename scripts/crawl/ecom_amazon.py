"""Amazon product page crawler."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote_plus

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import AMAZON_CONFIG, CrawlConfig
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)

# Search keywords per category
AMAZON_KEYWORDS: dict[str, list[str]] = {
    "electronics": ["smartphone unlocked", "wireless earbuds", "tablet android", "smart watch", "portable speaker"],
    "computers": ["laptop", "mechanical keyboard", "monitor 4k", "SSD nvme", "USB-C hub"],
    "clothing": ["mens winter jacket", "womens running shoes", "casual dress", "hiking boots", "cotton t-shirt"],
    "books": ["bestselling fiction", "cookbook 2024", "science fiction novel", "history book", "self help book"],
    "home_kitchen": ["air fryer", "espresso machine", "blender vitamix", "knife set kitchen", "instant pot"],
    "toys": ["lego set", "board game family", "action figure", "puzzle adult", "rc car"],
    "sports": ["yoga mat", "adjustable dumbbell", "cycling shorts", "tennis racket", "camping tent"],
    "beauty": ["face moisturizer", "electric shaver", "hair dryer", "cologne men", "makeup palette"],
    "automotive": ["phone mount car", "dash cam", "tire inflator portable", "car vacuum cleaner", "led headlight bulb"],
    "garden_tools": ["garden hose expandable", "pruning shears", "electric lawn mower", "gardening gloves", "sprinkler system"],
}


class AmazonCrawler(BaseCrawler):
    """Amazon product page crawler with aggressive anti-bot handling."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        super().__init__(config or AMAZON_CONFIG)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Search Amazon, extract /dp/{ASIN} URLs."""
        keywords = AMAZON_KEYWORDS.get(category, [category.replace("_", " ")])
        urls: list[str] = []

        for kw in keywords:
            if len(urls) >= limit:
                break

            for pg in range(1, 4):  # up to 3 search pages
                if len(urls) >= limit:
                    break
                search_url = (
                    f"{self.config.base_url}/s"
                    f"?k={quote_plus(kw)}&page={pg}"
                )
                try:
                    await self._navigate_with_stealth(page, search_url)

                    # Extract product links
                    links = await page.eval_on_selector_all(
                        "a[href*='/dp/']",
                        "els => els.map(e => e.href)",
                    )
                    for link in links:
                        asin_url = self._extract_dp_url(link)
                        if asin_url and asin_url not in urls:
                            urls.append(asin_url)

                except Exception as e:
                    logger.warning(f"[amazon/{category}] Search page {pg} failed for '{kw}': {e}")
                    break  # likely blocked, try next keyword

                await self._delay()

        return urls[:limit]

    async def _navigate_with_stealth(self, page: Page, url: str) -> None:
        """Navigate with extra anti-bot measures."""
        await page.goto(url, timeout=self.config.timeout, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Try to dismiss cookie consent
        try:
            accept_btn = page.locator("#sp-cc-accept, [data-action='a-dropdown-button']")
            if await accept_btn.count() > 0:
                await accept_btn.first.click()
                await page.wait_for_timeout(1000)
        except Exception:
            pass

        # Scroll down to trigger lazy loading
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.5)")
        await page.wait_for_timeout(1000)
        await page.evaluate("window.scrollBy(0, window.innerHeight * 0.5)")
        await page.wait_for_timeout(1500)

    def validate_page(self, html: str, category: str) -> ValidationResult:
        """Amazon-specific: must have product info markers."""
        html_lower = html.lower()
        has_dp = "/dp/" in html or "asin" in html_lower
        has_price = "a-price" in html_lower or "priceblock" in html_lower
        if not has_dp:
            return ValidationResult(False, "not_product_page")
        if not has_price:
            return ValidationResult(False, "no_price_element")
        return ValidationResult(True)

    @staticmethod
    def _extract_dp_url(url: str) -> str | None:
        """Keep full product URL path (bare /dp/ASIN often 404s)."""
        m = re.search(r"/dp/([A-Z0-9]{10})", url)
        if not m:
            return None
        # Strip query params but keep the full path
        clean = url.split("?")[0].split("#")[0]
        # Remove /ref=... suffix
        ref_idx = clean.find("/ref=")
        if ref_idx > 0:
            clean = clean[:ref_idx]
        return clean


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = AmazonCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
