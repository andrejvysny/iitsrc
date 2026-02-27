"""Annotation pipeline: 2-model consensus with GPT-4o escalation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import litellm

from shared.metrics import hallucination_rate, schema_valid, values_match
from shared.preprocessing import clean_html, truncate_to_tokens
from shared.prompts import build_messages
from shared.schemas import get_schema, get_schema_field_names
from shared.utils import parse_json_safe, timer

logger = logging.getLogger(__name__)

# Suppress litellm verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

PRIMARY_MODEL = "openrouter/deepseek/deepseek-v3.2"
SECONDARY_MODEL = "openrouter/qwen/qwen-2.5-72b-instruct"
ESCALATION_MODEL = "gpt-4o"

# Max content tokens — bounded by Qwen2.5 72B (may route to 32K on OpenRouter)
# Reserve ~4K for prompt template, schema, and output tokens
MAX_CONTENT_TOKENS = 28_000


@dataclass
class ModelResult:
    model_id: str
    response: dict | None = None
    raw_output: str = ""
    execution_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate: float = 0.0
    error: str | None = None


@dataclass
class AnnotationResult:
    data: dict
    domain: str
    page_name: str
    source_url: str
    agreement_scores: dict = field(default_factory=dict)
    per_model: dict = field(default_factory=dict)
    escalated_fields: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    json_ld: dict = field(default_factory=dict)  # P5: raw JSON-LD for traceability
    source_text_length: int = 0
    total_execution_time: float = 0.0
    total_cost_estimate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "domain": self.domain,
            "page_name": self.page_name,
            "source_url": self.source_url,
            "agreement_scores": self.agreement_scores,
            "per_model": self.per_model,
            "escalated_fields": self.escalated_fields,
            "validation": self.validation,
            "json_ld": self.json_ld,
            "source_text_length": self.source_text_length,
            "total_execution_time": round(self.total_execution_time, 2),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
        }


class Annotator:
    """Two-model consensus annotator with escalation."""

    def __init__(self, domain: str, delay: float = 1.0) -> None:
        self.domain = domain
        self.schema = get_schema(domain)
        self.fields = get_schema_field_names(domain)
        self.delay = delay

    def _fit_content(self, content: str) -> str:
        """Truncate content to shared token budget (equal for all primary/secondary models)."""
        return truncate_to_tokens(content, MAX_CONTENT_TOKENS)

    def _call_model(self, model_id: str, content: str,
                    max_tokens: int = 2048) -> ModelResult:
        """Call a single model via litellm."""
        result = ModelResult(model_id=model_id)
        content = self._fit_content(content)
        messages = build_messages(self.schema, content)

        kwargs: dict = {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        # JSON mode for models that support it
        if model_id.startswith("gpt-") or "deepseek" in model_id or "qwen" in model_id:
            kwargs["response_format"] = {"type": "json_object"}

        # Ensure OpenRouter routes to providers supporting json_object mode
        if "openrouter/" in model_id:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["provider"] = {"require_parameters": True}

        try:
            with timer() as t:
                resp = litellm.completion(**kwargs)
            result.execution_time = t["elapsed"]

            choice = resp.choices[0].message.content or ""
            result.raw_output = choice
            result.response = parse_json_safe(choice)

            usage = resp.usage
            if usage:
                result.prompt_tokens = usage.prompt_tokens or 0
                result.completion_tokens = usage.completion_tokens or 0

            try:
                result.cost_estimate = litellm.completion_cost(
                    completion_response=resp
                )
            except Exception:
                result.cost_estimate = 0.0

        except Exception as e:
            result.error = str(e)
            logger.error("Model %s failed: %s", model_id, e)

        return result

    def _compute_consensus(
        self, results: list[ModelResult]
    ) -> tuple[dict, dict, list[str]]:
        """Field-level consensus from model results.

        Returns (consensus_data, agreement_scores, disagreement_fields).
        """
        valid = [r for r in results if r.response is not None]

        if len(valid) == 0:
            return {f: None for f in self.fields}, {f: 0.0 for f in self.fields}, list(self.fields)

        if len(valid) == 1:
            data = valid[0].response
            return data, {f: 1.0 for f in self.fields}, []

        r1, r2 = valid[0].response, valid[1].response
        consensus = {}
        scores = {}
        disagreements = []

        for f in self.fields:
            v1 = r1.get(f)
            v2 = r2.get(f)
            if values_match(v1, v2):
                # Prefer non-None value
                consensus[f] = v1 if v1 is not None else v2
                scores[f] = 1.0
            else:
                consensus[f] = v1  # placeholder, will be escalated
                scores[f] = 0.0
                disagreements.append(f)

        return consensus, scores, disagreements

    def _escalate_field(
        self, content: str, field_name: str, candidates: list
    ) -> tuple | None:
        """Ask GPT-4o to resolve a field disagreement."""
        # Truncate content for cost control (token-based, not char-based)
        truncated = truncate_to_tokens(content, 3000)

        prompt = (
            f"Two models extracted different values for the '{field_name}' field.\n"
            f"Candidate A: {json.dumps(candidates[0])}\n"
            f"Candidate B: {json.dumps(candidates[1])}\n\n"
            f"Source content (truncated):\n{truncated}\n\n"
            f"Which value is correct? Respond with ONLY a JSON object: "
            f'{{"value": <correct_value>}}'
        )

        try:
            with timer() as t:
                resp = litellm.completion(
                    model=ESCALATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
            raw = resp.choices[0].message.content or ""
            parsed = parse_json_safe(raw)

            cost = 0.0
            try:
                cost = litellm.completion_cost(completion_response=resp)
            except Exception:
                pass

            mr = ModelResult(
                model_id=ESCALATION_MODEL,
                response=parsed,
                raw_output=raw,
                execution_time=t["elapsed"],
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
                cost_estimate=cost,
            )

            if parsed and "value" in parsed:
                return parsed["value"], mr
            return None

        except Exception as e:
            logger.error("Escalation failed for %s: %s", field_name, e)
            return None

    def _retry_with_feedback(
        self, model_id: str, content: str, result: ModelResult, source_text: str
    ) -> ModelResult:
        """Retry extraction with grounding feedback for flagged fields."""
        from rapidfuzz import fuzz

        flagged = []
        for field_name, value in (result.response or {}).items():
            if value is None or isinstance(value, list):
                continue
            if fuzz.partial_ratio(str(value)[:200], source_text) < 85:
                flagged.append(field_name)

        if not flagged:
            return result

        logger.info("Retrying %s — flagged ungrounded fields: %s", model_id, flagged)
        feedback = (
            f"Re-check these fields against the source — they may not match: {flagged}. "
            "Extract exact text from source or return null if not found."
        )
        return self._call_model(model_id, content + "\n\n" + feedback)

    def _validate(self, data: dict, source_text: str) -> dict:
        """Run validation checks on extracted data."""
        is_schema_valid = schema_valid(data, self.schema)
        h_rate = hallucination_rate(data, source_text)

        issues = []
        # Domain-specific rules
        price = data.get("price")
        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
            issues.append("price must be > 0")

        rating = data.get("rating")
        if rating is not None and isinstance(rating, (int, float)):
            if rating < 0 or rating > 5:
                issues.append("rating must be 0-5")

        address = data.get("address")
        if address is not None and isinstance(address, str) and len(address) < 10:
            issues.append("address too short (<10 chars)")

        return {
            "schema_valid": is_schema_valid,
            "rules_passed": len(issues) == 0,
            "source_grounded": h_rate < 0.3,
            "hallucination_rate": round(h_rate, 3),
            "issues": issues,
        }

    def annotate_page(
        self, html: str, page_name: str, source_url: str = ""
    ) -> AnnotationResult:
        """Full annotation pipeline for one HTML page."""
        with timer() as total_t:
            # Clean HTML — returns (content, json_ld_dict); JSON-LD prepended as text
            content, json_ld = clean_html(html)
            source_text_length = len(content)

            # Call both models (content truncated inside _call_model)
            r1 = self._call_model(PRIMARY_MODEL, content)
            # Retry if hallucination rate > 0.3
            if r1.response and hallucination_rate(r1.response, content) > 0.3:
                time.sleep(self.delay)
                r1 = self._retry_with_feedback(PRIMARY_MODEL, content, r1, content)

            time.sleep(self.delay)
            r2 = self._call_model(SECONDARY_MODEL, content)
            if r2.response and hallucination_rate(r2.response, content) > 0.3:
                time.sleep(self.delay)
                r2 = self._retry_with_feedback(SECONDARY_MODEL, content, r2, content)

            # Consensus
            consensus, scores, disagreements = self._compute_consensus([r1, r2])

            # Escalate disagreements
            escalated_fields = []
            escalation_cost = 0.0
            escalation_time = 0.0
            for field_name in disagreements:
                v1 = r1.response.get(field_name) if r1.response else None
                v2 = r2.response.get(field_name) if r2.response else None
                time.sleep(self.delay)
                esc = self._escalate_field(content, field_name, [v1, v2])
                if esc is not None:
                    value, mr = esc
                    consensus[field_name] = value
                    scores[field_name] = 0.5  # resolved via escalation
                    escalation_cost += mr.cost_estimate
                    escalation_time += mr.execution_time
                    escalated_fields.append(field_name)

            # Validate against full cleaned content (not truncated)
            validation = self._validate(consensus, content)

            total_cost = r1.cost_estimate + r2.cost_estimate + escalation_cost

        # Build per_model dict for serialization
        def _model_dict(r: ModelResult) -> dict:
            short_name = r.model_id.split("/")[-1]
            return {
                "response": r.response,
                "raw_output": r.raw_output,
                "execution_time": round(r.execution_time, 2),
                "tokens": r.prompt_tokens + r.completion_tokens,
                "cost": round(r.cost_estimate, 6),
                "error": r.error,
            }

        per_model = {
            PRIMARY_MODEL: _model_dict(r1),
            SECONDARY_MODEL.split("/")[-1]: _model_dict(r2),
        }

        return AnnotationResult(
            data=consensus,
            domain=self.domain,
            page_name=page_name,
            source_url=source_url,
            agreement_scores=scores,
            per_model=per_model,
            escalated_fields=escalated_fields,
            validation=validation,
            json_ld=json_ld,
            source_text_length=source_text_length,
            total_execution_time=total_t["elapsed"],
            total_cost_estimate=total_cost,
        )
