"""Base crawler with Playwright stealth, retry, CAPTCHA detection."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

from scripts.crawl.config import CrawlConfig
from scripts.crawl.manifest import Manifest
from scripts.crawl.validators import ValidationResult, extract_title, validate_html

logger = logging.getLogger(__name__)

VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1280, "height": 720},
]

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


class BaseCrawler(ABC):
    """Abstract base for all site crawlers."""

    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.manifest = Manifest(config.output_dir)
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, target_per_category: int | None = None) -> None:
        """Main entry: discover URLs per category, download pages."""
        target = target_per_category or self.config.target_per_category
        site = self.config.site
        logger.info(f"[{site}] Starting crawl, target={target}/category")

        async with async_playwright() as pw:
            browser = await self._launch_browser(pw)
            context = await self._create_context(browser)
            stealth = Stealth()
            page = await context.new_page()
            await stealth.apply_stealth_async(page)

            for category in self.config.categories:
                await self._crawl_category(page, category, target)

            await browser.close()

        stats = self.manifest.stats()
        logger.info(f"[{site}] Done. Stats: {stats}")

    async def _crawl_category(
        self, page: Page, category: str, target: int
    ) -> None:
        """Crawl a single category up to target count."""
        site = self.config.site
        downloaded = self._count_existing(category)
        if downloaded >= target:
            logger.info(f"[{site}/{category}] Already have {downloaded}/{target}, skipping")
            return

        logger.info(f"[{site}/{category}] Have {downloaded}/{target}, discovering URLs...")

        try:
            urls = await self.discover_urls(page, category, target * 2)
        except Exception as e:
            logger.error(f"[{site}/{category}] Discovery failed: {e}")
            return

        logger.info(f"[{site}/{category}] Found {len(urls)} candidate URLs")
        idx = downloaded

        for url in urls:
            if idx >= target:
                break

            filename = self.config.filename(category, idx + 1)
            if self.manifest.is_downloaded(filename):
                idx += 1
                continue

            html = await self._fetch_with_retry(page, url)
            if html is None:
                self.manifest.record_fail(
                    url=url, filename=filename, site=site,
                    category=category, error="fetch_failed",
                )
                continue

            vr = self._validate(html, category)
            if not vr.valid:
                self.manifest.record_fail(
                    url=url, filename=filename, site=site,
                    category=category, error=vr.reason,
                )
                logger.debug(f"[{site}/{category}] Skipped {url}: {vr.reason}")
                continue

            if self.manifest.is_duplicate_content(html):
                logger.debug(f"[{site}/{category}] Duplicate content, skipping")
                continue

            # Save
            out_path = self.config.output_dir / filename
            out_path.write_text(html, encoding="utf-8")
            self.manifest.record_ok(
                url=url, filename=filename, site=site,
                category=category, html=html,
            )
            idx += 1
            title = extract_title(html)
            logger.info(f"[{site}/{category}] Saved {filename} ({title[:60]})")

            await self._delay()

        logger.info(f"[{site}/{category}] Finished with {idx}/{target}")

    async def _fetch_with_retry(self, page: Page, url: str) -> str | None:
        """Fetch page with retries."""
        for attempt in range(self.config.retries):
            try:
                html = await self._fetch_page(page, url)
                return html
            except Exception as e:
                logger.warning(
                    f"[{self.config.site}] Attempt {attempt + 1}/{self.config.retries} "
                    f"failed for {url}: {e}"
                )
                await self._delay()
        return None

    async def _fetch_page(self, page: Page, url: str) -> str:
        """Navigate to URL, wait for load, return HTML."""
        await page.goto(url, timeout=self.config.timeout, wait_until="domcontentloaded")
        # Settle time for JS rendering
        await page.wait_for_timeout(random.randint(2000, 5000))
        return await page.content()

    def _validate(self, html: str, category: str) -> ValidationResult:
        """Run validation chain."""
        domain_label = "ecommerce" if self.config.domain == "ecom" else "realestate"
        vr = validate_html(html, domain=domain_label, min_size=self.config.min_file_size)
        if not vr.valid:
            return vr
        return self.validate_page(html, category)

    async def _delay(self) -> None:
        """Random delay between requests."""
        wait = random.uniform(self.config.delay_min, self.config.delay_max)
        await asyncio.sleep(wait)

    async def _launch_browser(self, pw: any) -> Browser:
        return await pw.chromium.launch(headless=not self.config.headed)

    async def _create_context(self, browser: Browser) -> BrowserContext:
        vp = random.choice(VIEWPORTS)
        ua = random.choice(USER_AGENTS)
        ctx = await browser.new_context(
            viewport=vp,
            user_agent=ua,
            locale="en-US",
            extra_http_headers=self.config.extra_headers,
        )
        return ctx

    def _count_existing(self, category: str) -> int:
        """Count already-downloaded files for this category."""
        prefix = f"{self.config.domain}_{self.config.site}_{category}_"
        return sum(
            1 for f in self.config.output_dir.iterdir()
            if f.name.startswith(prefix) and f.suffix == ".html"
        )

    # --- Abstract methods for subclasses ---

    @abstractmethod
    async def discover_urls(
        self, page: Page, category: str, limit: int
    ) -> list[str]:
        """Discover product/listing URLs for a category. Return up to `limit` URLs."""
        ...

    def validate_page(self, html: str, category: str) -> ValidationResult:
        """Site-specific validation. Override if needed."""
        return ValidationResult(True)
