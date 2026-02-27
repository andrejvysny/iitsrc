"""Nehnutelnosti.sk real estate listing crawler."""

from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import (
    NEHNUTELNOSTI_CATEGORY_MAP,
    NEHNUTELNOSTI_CONFIG,
    CrawlConfig,
)
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)


class NehnutelnostiCrawler(BaseCrawler):
    """Nehnutelnosti.sk listing page crawler."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        super().__init__(config or NEHNUTELNOSTI_CONFIG)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Browse category pages, extract listing detail URLs."""
        path = NEHNUTELNOSTI_CATEGORY_MAP.get(category)
        if not path:
            logger.warning(f"[nehnutelnosti/{category}] No URL mapping, skipping")
            return []

        urls: list[str] = []
        base = self.config.base_url

        for pg in range(1, 20):  # up to 20 pages of listings
            if len(urls) >= limit:
                break

            # Pagination
            if "?" in path:
                listing_url = f"{base}{path}&p={pg}"
            else:
                listing_url = f"{base}{path}?p={pg}"

            try:
                await page.goto(listing_url, timeout=self.config.timeout, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)

                # Accept cookies if shown
                await self._dismiss_cookies(page)

                # Extract detail links
                links = await page.eval_on_selector_all(
                    "a[href*='/detail/'], a[href*='/inzerat/']",
                    "els => els.map(e => e.href)",
                )

                if not links:
                    logger.debug(f"[nehnutelnosti/{category}] No links on page {pg}, stopping")
                    break

                for link in links:
                    clean = self._clean_url(link)
                    if clean and clean not in urls:
                        urls.append(clean)

            except Exception as e:
                logger.warning(f"[nehnutelnosti/{category}] Page {pg} failed: {e}")
                break

            await self._delay()

        return urls[:limit]

    def validate_page(self, html: str, category: str) -> ValidationResult:
        """Must contain property listing markers."""
        html_lower = html.lower()
        markers = ["cena", "plocha", "popis", "detail", "nehnutelnost"]
        found = sum(1 for m in markers if m in html_lower)
        if found < 2:
            return ValidationResult(False, "not_listing_page")
        return ValidationResult(True)

    @staticmethod
    async def _dismiss_cookies(page: Page) -> None:
        try:
            btn = page.locator(
                "button:has-text('Súhlasím'), "
                "button:has-text('Prijať'), "
                "button:has-text('Accept'), "
                "[id*='cookie'] button"
            )
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(500)
        except Exception:
            pass

    @staticmethod
    def _clean_url(url: str) -> str | None:
        """Remove tracking params, keep base URL."""
        if "/detail/" not in url and "/inzerat/" not in url:
            return None
        return url.split("?")[0].split("#")[0]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = NehnutelnostiCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
