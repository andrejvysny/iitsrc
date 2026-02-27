"""Zillow real estate listing crawler.

WARNING: Zillow uses PerimeterX anti-bot that blocks headless browsers.
Headed mode required, and may still be blocked. Consider Redfin/Homes.com
as fallback if consistently blocked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote_plus

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import ZILLOW_CONFIG, CrawlConfig
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)

# City-based search URLs (simpler = less likely to be blocked)
ZILLOW_SEARCHES: dict[str, list[str]] = {
    "house": ["New-York-NY", "Los-Angeles-CA", "Chicago-IL", "Houston-TX"],
    "apartment": ["New-York-NY", "San-Francisco-CA", "Miami-FL"],
    "condo": ["Miami-FL", "Seattle-WA", "Boston-MA"],
    "townhouse": ["Washington-DC", "Philadelphia-PA", "Denver-CO"],
    "land": ["Austin-TX", "Phoenix-AZ", "Jacksonville-FL"],
    "multifamily": ["New-York-NY", "Los-Angeles-CA", "Chicago-IL"],
    "luxury": ["Beverly-Hills-CA", "Manhattan-NY", "Aspen-CO"],
    "budget": ["Columbus-OH", "San-Antonio-TX", "Atlanta-GA"],
    "new_construction": ["Orlando-FL", "Dallas-TX", "Raleigh-NC"],
    "commercial": ["New-York-NY", "Chicago-IL", "Los-Angeles-CA"],
}

ZILLOW_TYPE_FILTER: dict[str, str] = {
    "house": "/type-single-family-home/",
    "apartment": "/type-apartment/",
    "condo": "/type-condo/",
    "townhouse": "/type-townhouse/",
    "land": "/type-lot-land/",
    "multifamily": "/type-multi-family/",
    "luxury": "/price-500000-0_price/",
    "budget": "/0-200000_price/",
    "new_construction": "/days-new-construction/",
    "commercial": "",
}


class ZillowCrawler(BaseCrawler):
    """Zillow listing page crawler. Requires headed mode due to PerimeterX."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        cfg = config or ZILLOW_CONFIG
        cfg.headed = True  # force headed mode
        super().__init__(cfg)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Search Zillow, extract /homedetails/ URLs."""
        cities = ZILLOW_SEARCHES.get(category, ["New-York-NY"])
        type_filter = ZILLOW_TYPE_FILTER.get(category, "")
        urls: list[str] = []

        for city in cities:
            if len(urls) >= limit:
                break

            search_url = f"{self.config.base_url}/{city.lower()}/{type_filter}"

            try:
                await page.goto(search_url, timeout=self.config.timeout, wait_until="domcontentloaded")
                await page.wait_for_timeout(5000)

                # Check for block
                html = await page.content()
                if self._is_blocked(html):
                    logger.warning(f"[zillow/{category}] Blocked by PerimeterX for {city}")
                    await self._delay()
                    continue

                # Scroll to load more results
                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, window.innerHeight)")
                    await page.wait_for_timeout(2000)

                # Extract listing links
                links = await page.eval_on_selector_all(
                    "a[href*='/homedetails/']",
                    "els => els.map(e => e.href)",
                )
                for link in links:
                    clean = self._clean_url(link)
                    if clean and clean not in urls:
                        urls.append(clean)

            except Exception as e:
                logger.warning(f"[zillow/{category}] Search failed for {city}: {e}")

            await self._delay()

        if not urls:
            logger.warning(
                f"[zillow/{category}] No URLs found. Zillow may be blocking. "
                "Consider using headed mode or a fallback site."
            )

        return urls[:limit]

    def validate_page(self, html: str, category: str) -> ValidationResult:
        html_lower = html.lower()
        if self._is_blocked(html):
            return ValidationResult(False, "blocked_perimeterx")
        markers = ["homedetails", "zestimate", "price", "beds", "baths", "sqft", "zillow"]
        found = sum(1 for m in markers if m in html_lower)
        if found < 3:
            return ValidationResult(False, "not_listing_page")
        return ValidationResult(True)

    @staticmethod
    def _is_blocked(html: str) -> bool:
        html_lower = html.lower()
        return "access to this page has been denied" in html_lower or (
            len(html) < 15_000 and "perimeterx" in html_lower
        )

    @staticmethod
    def _clean_url(url: str) -> str | None:
        m = re.search(r"(https://www\.zillow\.com/homedetails/[^?&#]+)", url)
        return m.group(1) if m else None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = ZillowCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
