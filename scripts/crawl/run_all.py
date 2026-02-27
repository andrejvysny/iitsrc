"""Orchestrator: run all or selected site crawlers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from scripts.crawl.config import ALL_CONFIGS
from scripts.crawl.ecom_amazon import AmazonCrawler
from scripts.crawl.ecom_ebay import EbayCrawler
from scripts.crawl.re_nehnutelnosti import NehnutelnostiCrawler
from scripts.crawl.re_realtor import RealtorCrawler
from scripts.crawl.re_reality import RealityCrawler
from scripts.crawl.re_zillow import ZillowCrawler

CRAWLER_MAP = {
    "amazon": AmazonCrawler,
    "ebay": EbayCrawler,
    # RE crawlers — kept but not in default set (blocked by anti-bot)
    "zillow": ZillowCrawler,
    "realtor": RealtorCrawler,
    "nehnutelnosti": NehnutelnostiCrawler,
    "reality": RealityCrawler,
}

# Default set: only reliably working crawlers
DEFAULT_SITES = ["ebay", "amazon"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data collection crawlers")
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=list(CRAWLER_MAP.keys()),
        default=DEFAULT_SITES,
        help="Sites to crawl (default: ebay, amazon)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Override target pages per category",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def run_site(site: str, target: int | None) -> None:
    """Run a single site crawler."""
    cls = CRAWLER_MAP[site]
    crawler = cls()
    await crawler.run(target_per_category=target)


async def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Crawling sites: {', '.join(args.sites)}")

    for site in args.sites:
        logger.info(f"--- Starting {site} ---")
        try:
            await run_site(site, args.target)
        except Exception as e:
            logger.error(f"[{site}] Crawler failed: {e}")
            continue
        logger.info(f"--- Finished {site} ---")

    logger.info("All done.")


if __name__ == "__main__":
    asyncio.run(main())
