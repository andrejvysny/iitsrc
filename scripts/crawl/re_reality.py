"""Reality.sk real estate listing crawler."""

from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import Page

from scripts.crawl.base import BaseCrawler
from scripts.crawl.config import (
    REALITY_CATEGORY_MAP,
    REALITY_CONFIG,
    CrawlConfig,
)
from scripts.crawl.validators import ValidationResult

logger = logging.getLogger(__name__)

# Detail URL pattern: /{type}/{slug}/{hashID} where hashID is 6-12 alphanumeric chars
DETAIL_RE = re.compile(
    r"https://www\.reality\.sk/[^/]+/[^/]+-[^/]+/([A-Za-z0-9_-]{6,15})$"
)


class RealityCrawler(BaseCrawler):
    """Reality.sk listing page crawler."""

    def __init__(self, config: CrawlConfig | None = None) -> None:
        super().__init__(config or REALITY_CONFIG)

    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Browse category listing pages, extract detail URLs."""
        path = REALITY_CATEGORY_MAP.get(category)
        if not path:
            logger.warning(f"[reality/{category}] No URL mapping, skipping")
            return []

        urls: list[str] = []
        base = self.config.base_url

        for pg in range(1, 20):
            if len(urls) >= limit:
                break

            if "?" in path:
                listing_url = f"{base}{path}&strana={pg}"
            else:
                listing_url = f"{base}{path}?strana={pg}"

            try:
                await page.goto(listing_url, timeout=self.config.timeout, wait_until="domcontentloaded")
                await page.wait_for_timeout(3000)

                await self._dismiss_cookies(page)

                # Extract all links from the page
                all_links = await page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.href)",
                )

                found_on_page = 0
                for link in all_links:
                    clean = self._extract_detail_url(link)
                    if clean and clean not in urls:
                        urls.append(clean)
                        found_on_page += 1

                if found_on_page == 0:
                    logger.debug(f"[reality/{category}] No detail links on page {pg}, stopping")
                    break

                logger.debug(f"[reality/{category}] Page {pg}: {found_on_page} new URLs")

            except Exception as e:
                logger.warning(f"[reality/{category}] Page {pg} failed: {e}")
                break

            await self._delay()

        return urls[:limit]

    def validate_page(self, html: str, category: str) -> ValidationResult:
        """Must contain property listing markers."""
        html_lower = html.lower()
        markers = ["cena", "plocha", "popis", "detail", "reality"]
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
    def _extract_detail_url(url: str) -> str | None:
        """Extract listing detail URL with hash ID."""
        clean = url.split("?")[0].split("#")[0]
        if DETAIL_RE.match(clean):
            return clean
        return None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = RealityCrawler()
    await crawler.run()


if __name__ == "__main__":
    asyncio.run(main())
