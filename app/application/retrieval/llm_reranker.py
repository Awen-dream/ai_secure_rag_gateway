from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.rerankers import RetrievalReranker, sort_by_score
from app.infrastructure.llm.openai_client import OpenAIClient


_RANK_ID_RE = re.compile(r"R(\d+)")
_RANK_REASON_RE = re.compile(r"R(\d+)(?:\s*[:|\-]\s*(.+))?$", re.IGNORECASE)


@dataclass(frozen=True)
class LLMRerankDecision:
    order: list[int] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMReranker(RetrievalReranker):
    """Use an LLM to rerank top retrieval candidates when a remote model is configured."""

    client: OpenAIClient
    top_n: int = 8

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        ranked = sort_by_score(results)
        if not ranked or not self.client.can_execute():
            return ranked

        window = ranked[: self.top_n]
        instructions, input_text = self._build_prompt(query, window)
        response = self.client.generate_response(instructions=instructions, input_text=input_text)
        decision = self._parse_decision(response, len(window))
        if not decision.order:
            return ranked

        selected = [window[index] for index in decision.order if 0 <= index < len(window)]
        if not selected:
            return ranked

        seen_ids = {item.chunk.id for item in selected}
        top_score = max((item.score for item in window), default=0.0)
        reranked_window: list[RetrievalResult] = []
        for position, item in enumerate(selected):
            copy = item.model_copy(deep=True)
            copy.score = round(top_score + 0.2 - (position * 0.01), 4)
            copy.rerank_source = "llm"
            reason = decision.reasons.get(decision.order[position], "").strip()
            if reason:
                copy.rerank_notes = [reason]
            reranked_window.append(copy)

        for item in window:
            if item.chunk.id in seen_ids:
                continue
            reranked_window.append(item)

        tail = ranked[self.top_n :]
        return sort_by_score(reranked_window) + tail

    @staticmethod
    def _build_prompt(query: str, results: list[RetrievalResult]) -> tuple[str, str]:
        instructions = (
            "You are reranking enterprise retrieval candidates. "
            "Prefer candidates that directly answer the query, preserve exact policy wording, and have stronger citation usefulness. "
            "Return one candidate per line in best-first order using the format R1 | short_reason. "
            "Keep each reason under 12 words and do not add any prose outside the ranked lines."
        )
        lines = [f"Query: {query}", "Candidates:"]
        for index, result in enumerate(results, start=1):
            lines.extend(
                [
                    f"R{index} | title={result.document.title} | section={result.chunk.section_name} | score={result.score:.4f}",
                    f"text={result.chunk.text[:280].replace(chr(10), ' ')}",
                ]
            )
        return instructions, "\n".join(lines)

    @staticmethod
    def _parse_decision(response: str, limit: int) -> LLMRerankDecision:
        order: list[int] = []
        reasons: dict[int, str] = {}
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = _RANK_REASON_RE.match(stripped)
            if not match:
                continue
            index = int(match.group(1)) - 1
            if not (0 <= index < limit) or index in order:
                continue
            order.append(index)
            if match.group(2):
                reasons[index] = match.group(2).strip()
        if not order:
            for match in _RANK_ID_RE.findall(response.upper()):
                index = int(match) - 1
                if 0 <= index < limit and index not in order:
                    order.append(index)
        return LLMRerankDecision(order=order, reasons=reasons)
