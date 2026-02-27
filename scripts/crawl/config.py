"""Per-site crawl configuration and category definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DATA_DIR = Path(__file__).resolve().parents[2] / "idea-b-schema-pruning" / "data" / "raw_html"

ECOM_CATEGORIES: list[str] = [
    "electronics",
    "computers",
    "clothing",
    "books",
    "home_kitchen",
    "toys",
    "sports",
    "beauty",
    "automotive",
    "garden_tools",
]

RE_CATEGORIES: list[str] = [
    "house",
    "apartment",
    "condo",
    "townhouse",
    "land",
    "multifamily",
    "luxury",
    "budget",
    "new_construction",
    "commercial",
]

# Slovak category URL mappings
NEHNUTELNOSTI_CATEGORY_MAP: dict[str, str] = {
    "house": "/domy/predaj/",
    "apartment": "/byty/predaj/",
    "condo": "/byty/predaj/?type=apartman",  # closest match
    "townhouse": "/domy/predaj/?type=radovy",  # row houses
    "land": "/pozemky/predaj/",
    "multifamily": "/domy/predaj/?type=viacgeneracny",
    "luxury": "/nehnutelnosti/predaj/?cena-od=300000",
    "budget": "/nehnutelnosti/predaj/?cena-do=80000",
    "new_construction": "/novostavby/predaj/",
    "commercial": "/komercne-nehnutelnosti/predaj/",
}

REALITY_CATEGORY_MAP: dict[str, str] = {
    "house": "/domy/predaj/",
    "apartment": "/byty/predaj/",
    "condo": "/byty/apartman/predaj/",
    "townhouse": "/domy/predaj/",  # no direct equivalent
    "land": "/pozemky/predaj/",
    "multifamily": "/domy/predaj/",  # no direct equivalent
    "luxury": "/nehnutelnosti/predaj/?cena-od=300000",
    "budget": "/nehnutelnosti/predaj/?cena-do=80000",
    "new_construction": "/novostavby/predaj/",
    "commercial": "/komercne-objekty/predaj/",
}


@dataclass
class CrawlConfig:
    """Configuration for a single site crawler."""

    site: str  # e.g. "amazon", "ebay", "zillow"
    domain: str  # "ecommerce" or "realestate"
    base_url: str
    delay_min: float = 3.0
    delay_max: float = 7.0
    min_file_size: int = 5_000  # bytes
    timeout: int = 45_000  # ms
    retries: int = 3
    headed: bool = False  # run browser in headed mode
    target_per_category: int = 50
    categories: list[str] = field(default_factory=list)
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return BASE_DATA_DIR

    def filename(self, category: str, index: int) -> str:
        return f"{self.domain}_{self.site}_{category}_{index:03d}.html"


# Pre-built configs
AMAZON_CONFIG = CrawlConfig(
    site="amazon",
    domain="ecom",
    base_url="https://www.amazon.com",
    delay_min=5.0,
    delay_max=10.0,
    target_per_category=50,
    categories=ECOM_CATEGORIES,
)

EBAY_CONFIG = CrawlConfig(
    site="ebay",
    domain="ecom",
    base_url="https://www.ebay.com",
    delay_min=3.0,
    delay_max=7.0,
    target_per_category=50,
    categories=ECOM_CATEGORIES,
)

ZILLOW_CONFIG = CrawlConfig(
    site="zillow",
    domain="realestate",
    base_url="https://www.zillow.com",
    delay_min=5.0,
    delay_max=12.0,
    headed=True,
    target_per_category=25,
    categories=RE_CATEGORIES,
)

REALTOR_CONFIG = CrawlConfig(
    site="realtor",
    domain="realestate",
    base_url="https://www.realtor.com",
    delay_min=4.0,
    delay_max=8.0,
    target_per_category=25,
    categories=RE_CATEGORIES,
)

NEHNUTELNOSTI_CONFIG = CrawlConfig(
    site="nehnutelnosti",
    domain="realestate",
    base_url="https://www.nehnutelnosti.sk",
    delay_min=3.0,
    delay_max=6.0,
    target_per_category=25,
    categories=RE_CATEGORIES,
    extra_headers={"Accept-Language": "sk-SK,sk;q=0.9"},
)

REALITY_CONFIG = CrawlConfig(
    site="reality",
    domain="realestate",
    base_url="https://www.reality.sk",
    delay_min=3.0,
    delay_max=6.0,
    target_per_category=25,
    categories=RE_CATEGORIES,
    extra_headers={"Accept-Language": "sk-SK,sk;q=0.9"},
)

ALL_CONFIGS: dict[str, CrawlConfig] = {
    "amazon": AMAZON_CONFIG,
    "ebay": EBAY_CONFIG,
    "zillow": ZILLOW_CONFIG,
    "realtor": REALTOR_CONFIG,
    "nehnutelnosti": NEHNUTELNOSTI_CONFIG,
    "reality": REALITY_CONFIG,
}
