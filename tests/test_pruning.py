"""Tests for pruning strategies and schema keyword generation."""

import json
import pytest

from shared.schemas import generate_schema_keywords

# Import pruning via sys.path manipulation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "idea-b-schema-pruning" / "src"))

from pruning import (
    apply_pruning,
    convert_format,
    prune_none,
    prune_generic,
    prune_heuristic,
    STRATEGIES,
    FORMATS,
)


SAMPLE_HTML = """
<html>
<head><title>Test Page</title><script>var x=1;</script><style>.a{color:red}</style></head>
<body>
<nav><a href="/">Home</a></nav>
<main>
<h1>Product Name</h1>
<div class="content">
<p>This is a great product with amazing features.</p>
<div class="price">$29.99</div>
<div class="description">Detailed product description here.</div>
<table><tr><td>Weight</td><td>2.5kg</td></tr></table>
</div>
</main>
<footer>Copyright 2024</footer>
<div id="cookie-consent">Accept cookies</div>
</body>
</html>
"""

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string", "description": "Name of the product"},
        "price": {"type": "number", "description": "Product price"},
        "description": {"type": "string", "description": "Product description text"},
        "weight": {"type": "string", "description": "Product weight"},
    },
}


class TestGenerateSchemaKeywords:
    def test_extracts_property_names(self):
        kw = generate_schema_keywords(SAMPLE_SCHEMA)
        assert "product" in kw
        assert "name" in kw
        assert "price" in kw
        assert "description" in kw
        assert "weight" in kw

    def test_extracts_description_words(self):
        kw = generate_schema_keywords(SAMPLE_SCHEMA)
        assert "product" in kw
        assert "text" in kw

    def test_extracts_enum_values(self):
        schema = {
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                },
            }
        }
        kw = generate_schema_keywords(schema)
        assert "active" in kw
        assert "inactive" in kw
        assert "pending" in kw

    def test_recurses_nested(self):
        schema = {
            "properties": {
                "details": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string", "description": "Item color"},
                    },
                },
            }
        }
        kw = generate_schema_keywords(schema)
        assert "color" in kw
        assert "item" in kw

    def test_empty_schema(self):
        kw = generate_schema_keywords({})
        assert len(kw) == 0

    def test_filters_short_words(self):
        schema = {"properties": {"id": {"type": "string"}}}
        kw = generate_schema_keywords(schema)
        # "id" is only 2 chars, should be filtered
        assert "id" not in kw


class TestPruneNone:
    def test_returns_unchanged(self):
        result = prune_none(SAMPLE_HTML)
        assert result == SAMPLE_HTML


class TestPruneGeneric:
    def test_removes_script_style(self):
        result = prune_generic(SAMPLE_HTML)
        assert "<script>" not in result
        assert "<style>" not in result

    def test_removes_nav_footer(self):
        result = prune_generic(SAMPLE_HTML)
        assert "<nav>" not in result
        assert "<footer>" not in result

    def test_removes_cookie_consent(self):
        result = prune_generic(SAMPLE_HTML)
        assert "cookie-consent" not in result

    def test_keeps_main_content(self):
        result = prune_generic(SAMPLE_HTML)
        assert "Product Name" in result
        assert "$29.99" in result

    def test_nonempty_output(self):
        result = prune_generic(SAMPLE_HTML)
        assert len(result) > 0


class TestPruneHeuristic:
    def test_nonempty_output(self):
        result = prune_heuristic(SAMPLE_HTML, schema=SAMPLE_SCHEMA)
        assert len(result) > 0

    def test_keeps_relevant_content(self):
        result = prune_heuristic(SAMPLE_HTML, schema=SAMPLE_SCHEMA)
        assert "Product Name" in result


class TestApplyPruning:
    def test_none_strategy(self):
        result = apply_pruning(SAMPLE_HTML, "none")
        assert result == SAMPLE_HTML

    def test_generic_strategy(self):
        result = apply_pruning(SAMPLE_HTML, "generic")
        assert "<script>" not in result

    def test_heuristic_requires_schema(self):
        with pytest.raises(ValueError, match="requires a schema"):
            apply_pruning(SAMPLE_HTML, "heuristic")

    def test_semantic_requires_schema(self):
        with pytest.raises(ValueError, match="requires a schema"):
            apply_pruning(SAMPLE_HTML, "semantic")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            apply_pruning(SAMPLE_HTML, "nonexistent")


class TestConvertFormat:
    def test_simplified_html(self):
        result = convert_format(SAMPLE_HTML, "simplified_html")
        assert result == SAMPLE_HTML

    def test_markdown(self):
        result = convert_format(SAMPLE_HTML, "markdown")
        assert len(result) > 0
        assert "<html>" not in result

    def test_flat_json(self):
        result = convert_format(SAMPLE_HTML, "flat_json")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            convert_format(SAMPLE_HTML, "nonexistent")
