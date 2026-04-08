from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.application.query.intent import classify_query_intent_details
from app.application.query.rewrite import rewrite_query
from app.infrastructure.llm.openai_client import OpenAIClient

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ALLOWED_INTENTS = {"exact_lookup", "summary", "standard_qa"}


@dataclass(frozen=True)
class QueryUnderstandingResult:
    rewritten_query: str
    intent: str
    confidence: float
    reasons: list[str]
    source: str


class QueryUnderstandingService:
    """Hybrid query understanding with deterministic fallback and optional LLM refinement."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        self.openai_client = openai_client

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
            response_text = self.openai_client.generate_response(
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
            )

    def _fallback_result(
        self,
        query: str,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> QueryUnderstandingResult:
        rewritten_query = rewrite_query(query)
        if last_user_query:
            rewritten_query = rewrite_query(f"{last_user_query} {query}")
        intent_result = classify_query_intent_details(rewritten_query)
        return QueryUnderstandingResult(
            rewritten_query=rewritten_query,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            reasons=intent_result.reasons,
            source="rule",
        )

    def _should_use_llm(
        self,
        query: str,
        fallback: QueryUnderstandingResult,
        last_user_query: str | None = None,
        session_summary: str | None = None,
    ) -> bool:
        if not self.openai_client or not self.openai_client.can_execute():
            return False

        normalized = rewrite_query(query)
        if last_user_query or session_summary:
            return True
        if len(normalized) <= 18:
            return True
        if fallback.intent == "summary":
            return True
        if fallback.confidence < 0.8:
            return True
        return normalized != fallback.rewritten_query

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
        payload = self._extract_json_payload(response_text)
        rewritten_query = rewrite_query(str(payload.get("rewritten_query") or "").strip()) or fallback.rewritten_query
        intent = str(payload.get("intent") or "").strip()
        if intent not in _ALLOWED_INTENTS:
            raise ValueError("Unsupported query intent returned by LLM")

        confidence = self._normalize_confidence(payload.get("confidence"), fallback.confidence)
        reasons = self._normalize_reasons(payload.get("reasons"), fallback.reasons)

        # Guard against over-aggressive rewrites that erase the original query.
        if len(rewritten_query) < max(4, len(rewrite_query(original_query)) // 3):
            rewritten_query = fallback.rewritten_query
            reasons = list(dict.fromkeys(reasons + ["rewrite_guardrail"]))

        return QueryUnderstandingResult(
            rewritten_query=rewritten_query,
            intent=intent,
            confidence=confidence,
            reasons=list(dict.fromkeys(reasons + ["llm_query_understanding"])),
            source="llm",
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
