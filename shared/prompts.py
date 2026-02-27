"""Prompt templates for structured data extraction."""

import json

SYSTEM_PROMPT = (
    "You are a structured data extraction system. "
    "Extract data from web content into the JSON schema below.\n\n"
    "Rules:\n"
    "1. Extract values EXACTLY as they appear in the source — no paraphrasing or summarizing.\n"
    "2. If a field is not present in the source, return null.\n"
    "3. Do not invent, infer, or hallucinate any values.\n"
    "4. Output ONLY valid JSON matching the schema."
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
