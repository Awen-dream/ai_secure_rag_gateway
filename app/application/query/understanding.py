from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from app.application.query.intent import classify_query_intent_details
from app.application.query.rewrite import rewrite_query
from app.infrastructure.llm.base import LLMClient

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ALLOWED_INTENTS = {"exact_lookup", "summary", "standard_qa"}
_FOLLOW_UP_MARKER_RE = re.compile(r"(?:这个|那个|它|上面的|继续|刚才|上一轮|前面|呢[？?]?)")
_VERSION_RE = re.compile(r"(?:第\s*\d+\s*版|\bv\s*\d+(?:\.\d+)*\b|\bversion\s*\d+(?:\.\d+)*\b)", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(
    r"(?:\b[A-Z]{2,}-\d+\b|\b[a-z]+[_-]id\b|\bdoc[_-]?\d+\b|\b[a-z0-9]{6,}[-_][a-z0-9]{2,}\b)",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]")


class _LLMUnderstandingPayload(BaseModel):
    rewritten_query: str = Field(min_length=1, max_length=512)
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list, max_length=8)


@dataclass(frozen=True)
class QueryUnderstandingResult:
    rewritten_query: str
    intent: str
    confidence: float
    reasons: list[str]
    source: str
    rule_rewritten_query: str
    rule_intent: str
    rule_confidence: float
    rule_reasons: list[str]


class QueryUnderstandingService:
    """Hybrid query understanding with deterministic fallback and optional LLM refinement."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def understand(
        self,
        query: str,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> QueryUnderstandingResult:
        fallback = self._fallback_result(
            query,
            last_user_query=last_user_query,
            session_summary=session_summary,
        )
        if not self._should_use_llm(
            query,
            fallback,
            last_user_query=last_user_query,
            session_summary=session_summary,
        ):
            return fallback

        try:
            response_text = self.llm_client.generate_response(
                instructions=self._build_instructions(),
                input_text=self._build_input(
                    query,
                    fallback,
                    last_user_query=last_user_query,
                    session_summary=session_summary,
                ),
            )
            return self._parse_llm_result(response_text, query, fallback)
        except Exception:
            return QueryUnderstandingResult(
                rewritten_query=fallback.rewritten_query,
                intent=fallback.intent,
                confidence=fallback.confidence,
                reasons=list(dict.fromkeys(fallback.reasons + ["llm_fallback"])),
                source="fallback",
                rule_rewritten_query=fallback.rule_rewritten_query,
                rule_intent=fallback.rule_intent,
                rule_confidence=fallback.rule_confidence,
                rule_reasons=fallback.rule_reasons,
            )

    def _fallback_result(
        self,
        query: str,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> QueryUnderstandingResult:
        rewritten_query = rewrite_query(
            query,
            last_user_query=last_user_query,
            session_summary=session_summary,
        )
        intent_result = classify_query_intent_details(rewritten_query)
        return QueryUnderstandingResult(
            rewritten_query=rewritten_query,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            reasons=intent_result.reasons,
            source="rule",
            rule_rewritten_query=rewritten_query,
            rule_intent=intent_result.intent,
            rule_confidence=intent_result.confidence,
            rule_reasons=intent_result.reasons,
        )

    def _should_use_llm(
        self,
        query: str,
        fallback: QueryUnderstandingResult,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> bool:
        if not self.llm_client or not self.llm_client.can_execute():
            return False

        normalized = rewrite_query(query)
        if last_user_query or session_summary:
            return True
        if fallback.intent == "summary":
            return True
        if _FOLLOW_UP_MARKER_RE.search(normalized):
            return True
        if fallback.confidence < 0.45:
            return True
        if len(normalized) <= 6:
            return True
        return False

    @staticmethod
    def _build_instructions() -> str:
        return (
            "You are a query-understanding component for enterprise retrieval.\n"
            "Return valid JSON only with keys: rewritten_query, intent, confidence, reasons.\n"
            "Allowed intents: exact_lookup, summary, standard_qa.\n"
            "Do not add facts, entities, document ids, or versions that are not present.\n"
            "Rewrite only to clarify the user's retrieval intent.\n"
            "confidence must be a number between 0 and 1.\n"
            "reasons must be a short list of snake_case labels."
        )

    @staticmethod
    def _build_input(
        query: str,
        fallback: QueryUnderstandingResult,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> str:
        lines = [
            f"original_query: {query}\n"
            f"fallback_rewritten_query: {fallback.rewritten_query}\n"
            f"fallback_intent: {fallback.intent}\n"
            f"fallback_confidence: {fallback.confidence}\n"
            f"fallback_reasons: {', '.join(fallback.reasons)}"
        ]
        if last_user_query:
            lines.append(f"last_user_query: {last_user_query}")
        if session_summary:
            lines.append(f"session_summary: {session_summary[:240]}")
        return "\n".join(lines)

    def _parse_llm_result(
        self,
        response_text: str,
        original_query: str,
        fallback: QueryUnderstandingResult,
    ) -> QueryUnderstandingResult:
        payload = self._validate_payload(self._extract_json_payload(response_text))
        rewritten_query = rewrite_query(payload.rewritten_query.strip()) or fallback.rewritten_query
        intent = payload.intent.strip()
        if intent not in _ALLOWED_INTENTS:
            raise ValueError("Unsupported query intent returned by LLM")

        confidence = self._normalize_confidence(payload.confidence, fallback.confidence)
        reasons = self._normalize_reasons(payload.reasons, fallback.reasons)
        rewritten_query, reasons = self._apply_rewrite_guardrails(
            original_query=original_query,
            rewritten_query=rewritten_query,
            fallback=fallback,
            reasons=reasons,
        )
        if confidence < 0.55:
            rewritten_query = fallback.rewritten_query
            intent = fallback.intent
            confidence = fallback.confidence
            reasons = list(dict.fromkeys(reasons + ["low_llm_confidence"]))

        return QueryUnderstandingResult(
            rewritten_query=rewritten_query,
            intent=intent,
            confidence=confidence,
            reasons=list(dict.fromkeys(reasons + ["llm_query_understanding"])),
            source="llm",
            rule_rewritten_query=fallback.rule_rewritten_query,
            rule_intent=fallback.rule_intent,
            rule_confidence=fallback.rule_confidence,
            rule_reasons=fallback.rule_reasons,
        )

    @staticmethod
    def _extract_json_payload(response_text: str) -> dict:
        stripped = response_text.strip()
        fenced = _JSON_BLOCK_RE.search(stripped)
        candidate = fenced.group(1) if fenced else stripped
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("Query understanding payload must be a JSON object")
        return payload

    @staticmethod
    def _validate_payload(payload: dict) -> _LLMUnderstandingPayload:
        try:
            if hasattr(_LLMUnderstandingPayload, "model_validate"):
                return _LLMUnderstandingPayload.model_validate(payload)
            return _LLMUnderstandingPayload.parse_obj(payload)
        except ValidationError as exc:
            raise ValueError("Query understanding payload failed schema validation") from exc

    @staticmethod
    def _normalize_confidence(value: object, default: float) -> float:
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(normalized, 1.0))

    @staticmethod
    def _normalize_reasons(value: object, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return list(default)
        reasons = [str(item).strip() for item in value if str(item).strip()]
        return reasons[:8] or list(default)

    def _apply_rewrite_guardrails(
        self,
        original_query: str,
        rewritten_query: str,
        fallback: QueryUnderstandingResult,
        reasons: list[str],
    ) -> tuple[str, list[str]]:
        normalized_original = rewrite_query(original_query)

        if len(rewritten_query) < max(4, len(normalized_original) // 3):
            return fallback.rewritten_query, list(dict.fromkeys(reasons + ["rewrite_too_short"]))
        if len(rewritten_query) > max(160, len(fallback.rewritten_query) * 3):
            return fallback.rewritten_query, list(dict.fromkeys(reasons + ["rewrite_too_long"]))

        missing_terms = [
            term
            for term in self._extract_critical_terms(normalized_original)
            if term.lower() not in rewritten_query.lower()
        ]
        if missing_terms:
            return fallback.rewritten_query, list(dict.fromkeys(reasons + ["missing_critical_terms"]))

        return rewritten_query, reasons

    @staticmethod
    def _extract_critical_terms(query: str) -> list[str]:
        terms: list[str] = []
        for pattern in (_VERSION_RE, _IDENTIFIER_RE, _QUOTED_RE):
            for match in pattern.finditer(query):
                if match.groups():
                    terms.extend(group.strip() for group in match.groups() if group and group.strip())
                else:
                    value = match.group(0).strip()
                    if value:
                        terms.append(value)
        return list(dict.fromkeys(terms))
