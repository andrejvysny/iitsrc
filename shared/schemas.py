"""JSON schemas for e-commerce and real estate data extraction."""

ECOM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Product name/title"},
        "price": {"type": "number", "description": "Product price as a number"},
        "currency": {
            "type": "string",
            "enum": ["USD", "EUR", "CZK", "GBP"],
            "description": "Price currency code",
        },
        "brand": {
            "type": ["string", "null"],
            "description": "Brand/manufacturer name",
        },
        "description": {"type": "string", "description": "Product description"},
        "rating": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 5,
            "description": "Average rating 0-5",
        },
        "availability": {
            "type": "string",
            "enum": ["in_stock", "out_of_stock", "unknown"],
            "description": "Stock availability status",
        },
        "specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            "description": "Product specifications as key-value pairs",
        },
    },
    "required": ["name", "price", "currency", "description", "availability"],
}

REALESTATE_SCHEMA = {
    "type": "object",
    "properties": {
        "address": {"type": "string", "description": "Full property address"},
        "price": {"type": "number", "description": "Listing price as a number"},
        "currency": {
            "type": "string",
            "enum": ["USD", "EUR", "GBP"],
            "description": "Price currency code",
        },
        "bedrooms": {
            "type": ["integer", "null"],
            "description": "Number of bedrooms",
        },
        "bathrooms": {
            "type": ["integer", "null"],
            "description": "Number of bathrooms",
        },
        "area_sqm": {
            "type": ["number", "null"],
            "description": "Living area in square meters",
        },
        "description": {"type": "string", "description": "Property description"},
        "type": {
            "type": "string",
            "enum": ["apartment", "house", "condo", "land", "unknown"],
            "description": "Property type",
        },
    },
    "required": ["address", "price", "currency", "description", "type"],
}

# Schema field keywords for heuristic pruning
ECOM_KEYWORDS = {
    "name", "title", "product", "price", "cost", "currency", "dollar", "euro",
    "brand", "manufacturer", "maker", "description", "detail", "about",
    "rating", "review", "star", "score", "availability", "stock", "available",
    "in stock", "out of stock", "spec", "specification", "feature", "attribute",
}

REALESTATE_KEYWORDS = {
    "address", "location", "street", "city", "zip", "price", "cost", "rent",
    "currency", "dollar", "euro", "bedroom", "bed", "bath", "bathroom",
    "area", "sqft", "square", "size", "description", "detail", "about",
    "type", "apartment", "house", "condo", "land", "property",
}

SCHEMA_KEYWORDS = {
    "ecommerce": ECOM_KEYWORDS,
    "realestate": REALESTATE_KEYWORDS,
}

def get_schema(domain: str) -> dict:
    """Get schema for domain ('ecommerce' or 'realestate')."""
    if domain == "ecommerce":
        return ECOM_SCHEMA
    elif domain == "realestate":
        return REALESTATE_SCHEMA
    else:
        raise ValueError(f"Unknown domain: {domain}")

def get_schema_field_names(domain: str) -> list[str]:
    """Get list of field names for a domain schema."""
    schema = get_schema(domain)
    return list(schema["properties"].keys())

def get_schema_description(domain: str) -> str:
    """Get human-readable schema description for prompts."""
    import json
    return json.dumps(get_schema(domain), indent=2)
