"""Tests for shared.dataset module."""

import json
import pytest
import pandas as pd

from shared.dataset import (
    filter_candidates,
    sample_pages,
    parse_schema,
    parse_response,
    _build_record,
    _count_schema_properties,
)


class TestParseSchema:
    def test_valid_json(self):
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = parse_schema(schema)
        assert result is not None
        assert result["type"] == "object"

    def test_code_block(self):
        schema = '```json\n{"type": "object"}\n```'
        result = parse_schema(schema)
        assert result == {"type": "object"}

    def test_brace_extraction(self):
        schema = 'Here is the schema: {"type": "object"} end'
        result = parse_schema(schema)
        assert result == {"type": "object"}

    def test_invalid(self):
        assert parse_schema("not json") is None
        assert parse_schema("") is None
        assert parse_schema(None) is None


class TestParseResponse:
    def test_valid(self):
        resp = '{"name": "Test", "price": 9.99}'
        result = parse_response(resp)
        assert result["name"] == "Test"
        assert result["price"] == 9.99


class TestCountSchemaProperties:
    def test_flat(self):
        schema = {"properties": {"a": {}, "b": {}, "c": {}}}
        assert _count_schema_properties(schema) == 3

    def test_nested(self):
        schema = {
            "properties": {
                "a": {"type": "object", "properties": {"x": {}, "y": {}}},
                "b": {},
            }
        }
        assert _count_schema_properties(schema) == 4

    def test_array_items(self):
        schema = {
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"k": {}, "v": {}}},
                },
            }
        }
        assert _count_schema_properties(schema) == 3


class TestFilterCandidates:
    def _make_df(self, n=10, **overrides):
        data = {
            "response_is_valid": [True] * n,
            "schema_depth": [3] * n,
            "schema_keys": [10] * n,
            "content": ["x" * 500] * n,
            "source": ["https://example.com"] * n,
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_passes_valid(self):
        df = self._make_df(5)
        result = filter_candidates(df)
        assert len(result) == 5

    def test_filters_invalid_response(self):
        df = self._make_df(5, response_is_valid=[True, False, True, False, True])
        result = filter_candidates(df)
        assert len(result) == 3

    def test_filters_schema_depth(self):
        df = self._make_df(3, schema_depth=[1, 3, 7])
        result = filter_candidates(df)
        assert len(result) == 1

    def test_filters_content_length(self):
        df = self._make_df(3, content=["short", "x" * 500, "x" * 200_000])
        result = filter_candidates(df)
        assert len(result) == 1

    def test_filters_source_url(self):
        df = self._make_df(3, source=["https://a.com", "ftp://b.com", ""])
        result = filter_candidates(df)
        assert len(result) == 1


class TestSamplePages:
    def test_returns_n(self):
        df = pd.DataFrame({
            "schema_complexity_score": range(100),
        })
        result = sample_pages(df, n=20, seed=42)
        assert len(result) == 20

    def test_returns_all_if_small(self):
        df = pd.DataFrame({"schema_complexity_score": [1, 2, 3]})
        result = sample_pages(df, n=10)
        assert len(result) == 3

    def test_deterministic(self):
        df = pd.DataFrame({"schema_complexity_score": range(100)})
        r1 = sample_pages(df, n=20, seed=42)
        r2 = sample_pages(df, n=20, seed=42)
        assert list(r1["schema_complexity_score"]) == list(r2["schema_complexity_score"])


class TestBuildRecord:
    def test_valid_record(self):
        row = pd.Series({
            "id": "test_001",
            "source": "https://example.com",
            "prompt": "Extract data",
            "schema": '{"type": "object", "properties": {"name": {"type": "string"}}}',
            "response": '{"name": "Test"}',
            "content": "Some content",
            "schema_depth": 2,
            "schema_keys": 1,
            "schema_complexity_score": 0.5,
        })
        rec = _build_record(row)
        assert rec is not None
        assert rec["id"] == "test_001"
        assert rec["schema"]["type"] == "object"
        assert rec["gold"]["name"] == "Test"

    def test_invalid_schema(self):
        row = pd.Series({
            "schema": "not json",
            "response": '{"name": "Test"}',
        })
        rec = _build_record(row)
        assert rec is None

    def test_invalid_response(self):
        row = pd.Series({
            "schema": '{"type": "object"}',
            "response": "not json",
        })
        rec = _build_record(row)
        assert rec is None
