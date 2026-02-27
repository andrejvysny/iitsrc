"""Schema-aware HTML pruning strategies and format converters."""

import json
import re
from typing import Any

from bs4 import BeautifulSoup, Comment, Tag
import markdownify

from shared.schemas import generate_schema_keywords

# Tags to remove in generic pruning
REMOVE_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "iframe", "noscript", "svg", "link", "meta",
}

# Boilerplate class/id patterns
BOILERPLATE_PATTERNS = re.compile(
    r"cookie|consent|gdpr|banner|popup|modal|overlay|"
    r"sidebar|widget|social|share|newsletter|subscribe|"
    r"advertisement|ad-|ads-|advert|promo",
    re.IGNORECASE,
)

# Attributes to strip
STRIP_ATTRS = re.compile(r"^(data-|on[a-z]|style$|class$)")


# ---------------------------------------------------------------------------
# Pruning strategies
# ---------------------------------------------------------------------------

def prune_none(html: str, **kwargs: Any) -> str:
    """No pruning — return raw HTML as-is."""
    return html


def prune_generic(html: str, **kwargs: Any) -> str:
    """Remove boilerplate: scripts, nav, ads, comments, noisy attrs."""
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted tags
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove boilerplate elements by id/class (collect first, then decompose)
    to_remove = []
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        tag_id = tag.get("id", "")
        if tag_id:
            if BOILERPLATE_PATTERNS.search(str(tag_id)):
                to_remove.append(tag)
                continue
        tag_class = tag.get("class", [])
        if tag_class and BOILERPLATE_PATTERNS.search(" ".join(tag_class)):
            to_remove.append(tag)
    for tag in to_remove:
        tag.decompose()

    # Strip noisy attributes
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        attrs_to_remove = [
            attr for attr in tag.attrs
            if STRIP_ATTRS.match(attr)
        ]
        for attr in attrs_to_remove:
            del tag[attr]

    return str(soup)


def prune_heuristic(
    html: str,
    schema: dict,
    threshold: float = 0.05,
    min_text_len: int = 200,
    **kwargs: Any,
) -> str:
    """Generic pruning + keyword overlap scoring → prune low-relevance subtrees."""
    # First apply generic pruning
    html = prune_generic(html)
    soup = BeautifulSoup(html, "lxml")

    keywords = generate_schema_keywords(schema)
    if not keywords:
        return str(soup)

    def _relevance(tag: Tag) -> float:
        """Compute keyword overlap ratio for a tag's text."""
        text = tag.get_text(separator=" ", strip=True).lower()
        if not text:
            return 0.0
        words = set(text.split())
        if not words:
            return 0.0
        overlap = words & keywords
        return len(overlap) / len(keywords)

    # Score and prune block-level elements
    block_tags = {"div", "section", "article", "main", "table", "ul", "ol", "dl", "p"}
    for tag in soup.find_all(block_tags):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) < min_text_len:
            continue
        if _relevance(tag) < threshold:
            tag.decompose()

    return str(soup)


def prune_semantic(
    html: str,
    schema: dict,
    threshold: float = 0.25,
    **kwargs: Any,
) -> str:
    """Generic pruning + embedding similarity → keep relevant subtrees."""
    html = prune_generic(html)
    soup = BeautifulSoup(html, "lxml")

    model = _get_embedding_model()

    # Build schema description text for embedding
    schema_texts = []
    for name, defn in schema.get("properties", {}).items():
        desc = defn.get("description", name) if isinstance(defn, dict) else name
        schema_texts.append(f"{name}: {desc}")
    if not schema_texts:
        return str(soup)

    schema_embeddings = model.encode(schema_texts, normalize_embeddings=True)

    # Collect text nodes from block elements
    block_tags = {"div", "section", "article", "main", "table", "ul", "ol", "dl", "p", "span", "td", "th", "li", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "h6"}
    nodes_to_check: list[tuple[Tag, str]] = []
    for tag in soup.find_all(block_tags):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 20:
            nodes_to_check.append((tag, text))

    if not nodes_to_check:
        return str(soup)

    # Embed node texts
    node_texts = [text for _, text in nodes_to_check]
    node_embeddings = model.encode(node_texts, normalize_embeddings=True)

    # Compute max cosine similarity to any schema field
    import numpy as np
    # shape: (n_nodes, n_schema_fields)
    similarities = node_embeddings @ schema_embeddings.T
    max_sims = np.max(similarities, axis=1)  # shape: (n_nodes,)

    # Mark nodes to keep
    keep_tags: set[int] = set()
    for i, (tag, _) in enumerate(nodes_to_check):
        if max_sims[i] >= threshold:
            # Keep this node and all ancestors
            keep_tags.add(id(tag))
            for parent in tag.parents:
                if isinstance(parent, Tag):
                    keep_tags.add(id(parent))

    # Prune nodes below threshold (only block-level, leaf-ish)
    for i, (tag, _) in enumerate(nodes_to_check):
        if id(tag) not in keep_tags and max_sims[i] < threshold:
            # Only decompose if no kept descendants
            has_kept_child = any(
                id(child) in keep_tags
                for child in tag.descendants
                if isinstance(child, Tag)
            )
            if not has_kept_child:
                tag.decompose()

    return str(soup)


# Lazy-loaded singleton for sentence-transformers model
_embedding_model = None


def _get_embedding_model():
    """Lazy-load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def to_simplified_html(html: str) -> str:
    """Return HTML as-is (simplified format = pruned HTML)."""
    return html


def to_markdown(html: str) -> str:
    """Convert HTML to markdown."""
    return markdownify.markdownify(html, strip=["img"])


def to_flat_json(html: str) -> str:
    """Convert HTML to flat JSON of sequential text blocks."""
    soup = BeautifulSoup(html, "lxml")
    blocks = []
    for tag in soup.find_all(True):
        # Only take direct text (not children's text)
        text = tag.string
        if text and text.strip():
            blocks.append({
                "tag": tag.name,
                "text": text.strip(),
            })
    # Fallback: if no direct strings, use get_text per block element
    if not blocks:
        for tag in soup.find_all(["p", "div", "span", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt"]):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                blocks.append({"tag": tag.name, "text": text})
    return json.dumps(blocks, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

STRATEGIES = {
    "none": prune_none,
    "generic": prune_generic,
    "heuristic": prune_heuristic,
    "semantic": prune_semantic,
}

FORMATS = {
    "simplified_html": to_simplified_html,
    "markdown": to_markdown,
    "flat_json": to_flat_json,
}


def apply_pruning(html: str, strategy: str, schema: dict | None = None, **kwargs: Any) -> str:
    """Apply a pruning strategy to HTML."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(STRATEGIES.keys())}")
    if strategy in ("heuristic", "semantic") and schema is None:
        raise ValueError(f"Strategy '{strategy}' requires a schema argument")
    kwargs["schema"] = schema
    return STRATEGIES[strategy](html, **kwargs)


def convert_format(html: str, fmt: str) -> str:
    """Convert pruned HTML to output format."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format: {fmt}. Choose from {list(FORMATS.keys())}")
    return FORMATS[fmt](html)
