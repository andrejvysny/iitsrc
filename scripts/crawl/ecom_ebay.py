"""eBay product page crawler."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote_plus

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import EBAY_CONFIG, CrawlConfig
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)

# Search keywords per category
EBAY_KEYWORDS: dict[str, list[str]] = {
    "electronics": ["smartphone", "wireless earbuds", "tablet", "smart watch", "bluetooth speaker"],
    "computers": ["laptop", "gaming keyboard", "monitor 27", "SSD 1TB", "webcam"],
    "clothing": ["mens jacket", "womens dress", "sneakers", "winter coat", "running shoes"],
    "books": ["bestseller novel", "cookbook", "science fiction book", "textbook", "biography"],
    "home_kitchen": ["air fryer", "coffee maker", "blender", "knife set", "instant pot"],
    "toys": ["lego set", "board game", "action figure", "puzzle 1000", "remote control car"],
    "sports": ["yoga mat", "dumbbell set", "cycling jersey", "tennis racket", "hiking backpack"],
    "beauty": ["moisturizer", "electric razor", "hair dryer", "perfume", "makeup kit"],
    "automotive": ["car phone mount", "dash cam", "tire inflator", "car vacuum", "LED headlight"],
    "garden_tools": ["garden hose", "pruning shears", "lawn mower", "garden gloves", "sprinkler"],
}


class EbayCrawler(BaseCrawler):
    """eBay Buy It Now product page crawler."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        super().__init__(config or EBAY_CONFIG)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Search eBay for Buy It Now listings, extract item URLs."""
        keywords = EBAY_KEYWORDS.get(category, [category.replace("_", " ")])
        urls: list[str] = []

        for kw in keywords:
            if len(urls) >= limit:
                break
            search_url = (
                f"{self.config.base_url}/sch/i.html"
                f"?_nkw={quote_plus(kw)}&LH_BIN=1&_pgn=1"
            )
            try:
                await page.goto(search_url, timeout=self.config.timeout, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # Extract item links from search results
                links = await page.eval_on_selector_all(
                    "a[href*='/itm/']",
                    "els => els.map(e => e.href)",
                )
                for link in links:
                    clean = self._clean_item_url(link)
                    if clean and clean not in urls:
                        urls.append(clean)

                # Try page 2
                if len(urls) < limit:
                    search_url_p2 = search_url.replace("_pgn=1", "_pgn=2")
                    await self._delay()
                    await page.goto(search_url_p2, timeout=self.config.timeout, wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)
                    links = await page.eval_on_selector_all(
                        "a[href*='/itm/']",
                        "els => els.map(e => e.href)",
                    )
                    for link in links:
                        clean = self._clean_item_url(link)
                        if clean and clean not in urls:
                            urls.append(clean)

            except Exception as e:
                logger.warning(f"[ebay/{category}] Search failed for '{kw}': {e}")

            await self._delay()

        return urls[:limit]

    def validate_page(self, html: str, category: str) -> ValidationResult:
        """eBay-specific: must have item ID in page."""
        if "/itm/" not in html and "item-id" not in html.lower():
            return ValidationResult(False, "not_item_page")
        return ValidationResult(True)

    @staticmethod
    def _clean_item_url(url: str) -> str | None:
        """Extract canonical /itm/{id} URL."""
        m = re.search(r"(https://www\.ebay\.com/itm/\d+)", url)
        if m:
            return m.group(1)
        m = re.search(r"(https://www\.ebay\.com/itm/[^?&#]+)", url)
        return m.group(1) if m else None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = EbayCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
