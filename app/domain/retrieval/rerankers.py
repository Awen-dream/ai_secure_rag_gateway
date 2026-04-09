from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass

from app.application.query.rewrite import build_query_rewrite_plan
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.retrievers import normalize_terms


def sort_by_score(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(results, key=lambda item: item.score, reverse=True)


def weighted_fusion(
    keyword_score: float,
    vector_score: float,
    keyword_weight: float,
    vector_weight: float,
    title_boost: float = 0.0,
) -> float:
    return (keyword_score * keyword_weight) + (vector_score * vector_weight) + title_boost


@dataclass(frozen=True)
class HeuristicReranker:
    """Local reranker that sharpens fused retrieval with cross-encoder-style matching signals."""

    mode: str = "heuristic"
    top_n: int = 8

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Boost coverage, proximity, recency and exact phrase matches on the top candidate window."""

        if self.mode == "disabled":
            return sort_by_score(results)

        plan = build_query_rewrite_plan(query)
        normalized_query = plan.rewritten_query.strip().lower()
        query_terms = list(dict.fromkeys(plan.expanded_terms or normalize_terms(normalized_query)))
        exact_phrases = [phrase.lower() for phrase in plan.exact_phrases if phrase.strip()]
        reranked: list[RetrievalResult] = []
        for index, result in enumerate(sort_by_score(results)):
            if index >= self.top_n:
                reranked.append(result)
                continue

            body = result.chunk.text.lower()
            title = result.document.title.lower()
            section = result.chunk.section_name.lower()
            joined = f"{title} {section} {body}"

            exact_phrase_boost = 0.0
            if normalized_query and normalized_query in joined:
                exact_phrase_boost += 0.12
            exact_phrase_boost += sum(0.06 for phrase in exact_phrases[:3] if phrase in joined)

            covered_terms = [term for term in query_terms if term in joined]
            coverage_ratio = (len(set(covered_terms)) / max(len(set(query_terms)), 1)) if query_terms else 0.0
            coverage_boost = min(coverage_ratio * 0.16, 0.16)
            title_term_boost = min(sum(0.025 for term in covered_terms if term in title), 0.12)
            section_boost = min(sum(0.02 for term in covered_terms if term in section), 0.08)
            density_boost = min(len(result.matched_terms) * 0.015, 0.1)
            proximity_boost = 0.08 if _terms_appear_in_order(query_terms[:6], joined) else 0.0
            recency_boost = _recency_boost(result.document.updated_at, plan.recency_hint)

            if self.mode == "cross-encoder-fallback":
                scale = 1.25
            else:
                scale = 1.0
            result.score = round(
                result.score
                + ((exact_phrase_boost + coverage_boost + title_term_boost + section_boost + density_boost + proximity_boost + recency_boost) * scale),
                4,
            )
            reranked.append(result)
        return sort_by_score(reranked)


def _terms_appear_in_order(terms: list[str], text: str) -> bool:
    if len(terms) < 2:
        return False
    cursor = 0
    matched = 0
    for term in terms:
        position = text.find(term, cursor)
        if position < 0:
            continue
        matched += 1
        cursor = position + len(term)
    return matched >= min(3, len(terms))


def _recency_boost(updated_at: datetime, recency_hint: bool) -> float:
    if not recency_hint:
        return 0.0
    age_days = max((datetime.utcnow() - updated_at).days, 0)
    if age_days <= 30:
        return 0.08
    if age_days <= 180:
        return 0.04
    if age_days <= 365:
        return 0.02
    return 0.0
