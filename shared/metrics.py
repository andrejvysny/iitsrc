"""Evaluation metrics for structured data extraction."""

import json
import time
from rapidfuzz import fuzz


def normalize_string(s: str) -> str:
    """Normalize string for comparison."""
    if not isinstance(s, str):
        return str(s).strip().lower()
    return s.strip().lower()


def values_match(pred, gold, fuzzy_threshold: float = 0.85) -> bool:
    """Check if two values match (exact or fuzzy for strings)."""
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(gold - pred) < 0.01 * max(abs(gold), 1)
    if isinstance(gold, str) and isinstance(pred, str):
        if normalize_string(pred) == normalize_string(gold):
            return True
        return fuzz.ratio(normalize_string(pred), normalize_string(gold)) / 100 >= fuzzy_threshold
    if isinstance(gold, list) and isinstance(pred, list):
        if len(gold) == 0 and len(pred) == 0:
            return True
        if len(gold) == 0 or len(pred) == 0:
            return False
        # For spec lists, compare as sets of key-value pairs
        try:
            gold_set = {(normalize_string(d["key"]), normalize_string(d["value"])) for d in gold}
            pred_set = {(normalize_string(d["key"]), normalize_string(d["value"])) for d in pred}
            if len(gold_set) == 0:
                return len(pred_set) == 0
            intersection = gold_set & pred_set
            return len(intersection) / len(gold_set) >= 0.5
        except (KeyError, TypeError):
            return False
    return str(pred) == str(gold)


def field_f1(predicted: dict, gold: dict) -> dict:
    """Compute per-field and macro F1, precision, recall."""
    all_fields = set(gold.keys()) | set(predicted.keys())
    tp = fp = fn = 0
    per_field = {}

    for field in all_fields:
        g = gold.get(field)
        p = predicted.get(field)

        if g is not None and p is not None and values_match(p, g):
            tp += 1
            per_field[field] = {"status": "tp", "pred": p, "gold": g}
        elif p is not None and (g is None or not values_match(p, g)):
            fp += 1
            if g is not None:
                fn += 1  # also missed the correct value
            per_field[field] = {"status": "fp", "pred": p, "gold": g}
        elif g is not None and p is None:
            fn += 1
            per_field[field] = {"status": "fn", "pred": None, "gold": g}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_field": per_field,
    }


def exact_match(predicted: dict, gold: dict) -> bool:
    """Check if prediction exactly matches gold."""
    return all(
        values_match(predicted.get(k), gold.get(k))
        for k in set(gold.keys()) | set(predicted.keys())
    )


def json_valid(output: str) -> bool:
    """Check if string is valid JSON."""
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def schema_valid(data: dict, schema: dict) -> bool:
    """Validate data against JSON schema."""
    import jsonschema
    try:
        jsonschema.validate(data, schema)
        return True
    except jsonschema.ValidationError:
        return False


def hallucination_rate(predicted: dict, source_text: str) -> float:
    """Fraction of predicted string values not found in source text."""
    if not predicted or not source_text:
        return 0.0
    source_lower = source_text.lower()
    total = 0
    hallucinated = 0
    for k, v in predicted.items():
        if v is None:
            continue
        if isinstance(v, str) and len(v) > 2:
            total += 1
            # Check if value or significant substring exists in source
            v_lower = v.lower()
            if v_lower not in source_lower:
                # Try first 50 chars for long descriptions
                if len(v_lower) > 50:
                    if v_lower[:50] not in source_lower:
                        hallucinated += 1
                else:
                    hallucinated += 1
        elif isinstance(v, (int, float)):
            total += 1
            if str(v) not in source_text and str(int(v)) not in source_text:
                hallucinated += 1
    return hallucinated / total if total > 0 else 0.0


def compute_all_metrics(predicted: dict, gold: dict, schema: dict,
                        source_text: str, raw_output: str) -> dict:
    """Compute all metrics for a single prediction."""
    f1_result = field_f1(predicted, gold)
    return {
        "f1": f1_result["macro_f1"],
        "precision": f1_result["precision"],
        "recall": f1_result["recall"],
        "exact_match": exact_match(predicted, gold),
        "json_valid": json_valid(raw_output),
        "schema_valid": schema_valid(predicted, schema),
        "hallucination_rate": hallucination_rate(predicted, source_text),
        "per_field": f1_result["per_field"],
    }
