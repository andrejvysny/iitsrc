"""Prompt templates for structured data extraction."""

import json

SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. Extract data from web content "
    "according to the JSON schema below. Output ONLY valid JSON. Use null for missing fields. "
    "Do not invent values not present in the content."
)

EXTRACTION_TEMPLATE = """{system}

JSON Schema:
{schema}

Web Content:
{content}

Output:"""


def build_extraction_prompt(schema: dict, content: str, include_system: bool = True) -> str:
    """Build extraction prompt with schema and content."""
    return EXTRACTION_TEMPLATE.format(
        system=SYSTEM_PROMPT if include_system else "",
        schema=json.dumps(schema, indent=2),
        content=content,
    ).strip()


def build_messages(schema: dict, content: str) -> list[dict]:
    """Build chat messages for API-based models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"JSON Schema:\n{json.dumps(schema, indent=2)}\n\nWeb Content:\n{content}\n\nOutput:",
        },
    ]
