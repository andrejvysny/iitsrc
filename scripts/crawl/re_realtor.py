"""Realtor.com real estate listing crawler.

WARNING: Realtor.com blocks headless browsers aggressively.
Headed mode required. Consider fallback sites if consistently blocked.
"""

from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import REALTOR_CONFIG, CrawlConfig
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)

REALTOR_TYPE_MAP: dict[str, str] = {
    "house": "type-single-family-home",
    "apartment": "type-apartment",
    "condo": "type-condo",
    "townhouse": "type-townhome",
    "land": "type-land",
    "multifamily": "type-multi-family-home",
    "luxury": "price-500000-na",
    "budget": "price-na-150000",
    "new_construction": "age-new",
    "commercial": "type-commercial",
}

REALTOR_CITIES: list[str] = [
    "New-York_NY",
    "Los-Angeles_CA",
    "Chicago_IL",
    "Houston_TX",
    "Phoenix_AZ",
    "Philadelphia_PA",
    "San-Antonio_TX",
    "San-Diego_CA",
    "Dallas_TX",
    "Austin_TX",
]


class RealtorCrawler(BaseCrawler):
    """Realtor.com listing page crawler. Requires headed mode."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        cfg = config or REALTOR_CONFIG
        cfg.headed = True  # force headed mode
        super().__init__(cfg)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Search realtor.com by city + type, extract listing detail URLs."""
        type_slug = REALTOR_TYPE_MAP.get(category, f"type-{category}")
        urls: list[str] = []

        for city in REALTOR_CITIES:
            if len(urls) >= limit:
                break

            for pg in range(1, 4):
                if len(urls) >= limit:
                    break

                search_url = (
                    f"{self.config.base_url}/realestateandhomes-search"
                    f"/{city}/{type_slug}/pg-{pg}"
                )

                try:
                    await page.goto(search_url, timeout=self.config.timeout, wait_until="domcontentloaded")
                    await page.wait_for_timeout(4000)

                    html = await page.content()
                    if self._is_blocked(html):
                        logger.warning(f"[realtor/{category}] Blocked for {city}")
                        break

                    # Extract listing detail links
                    links = await page.eval_on_selector_all(
                        "a[href*='/realestateandhomes-detail/']",
                        "els => els.map(e => e.href)",
                    )

                    if not links:
                        break

                    for link in links:
                        clean = self._clean_url(link)
                        if clean and clean not in urls:
                            urls.append(clean)

                except Exception as e:
                    logger.warning(f"[realtor/{category}] Search {city} pg {pg} failed: {e}")
                    break

                await self._delay()

        if not urls:
            logger.warning(
                f"[realtor/{category}] No URLs found. Realtor.com may be blocking. "
                "Consider headed mode or a fallback site."
            )

        return urls[:limit]

    def validate_page(self, html: str, category: str) -> ValidationResult:
        if self._is_blocked(html):
            return ValidationResult(False, "blocked")
        html_lower = html.lower()
        markers = ["realtor", "price", "bed", "bath", "sqft", "listing", "property"]
        found = sum(1 for m in markers if m in html_lower)
        if found < 3:
            return ValidationResult(False, "not_listing_page")
        return ValidationResult(True)

    @staticmethod
    def _is_blocked(html: str) -> bool:
        return len(html) < 5_000 and (
            "could not be processed" in html.lower()
            or "access denied" in html.lower()
        )

    @staticmethod
    def _clean_url(url: str) -> str | None:
        m = re.search(
            r"(https://www\.realtor\.com/realestateandhomes-detail/[^?&#]+)", url
        )
        return m.group(1) if m else None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = RealtorCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
