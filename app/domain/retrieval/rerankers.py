from __future__ import annotations

from dataclasses import dataclass

from app.domain.retrieval.models import RetrievalResult


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
    """Lightweight reranker that sharpens fused retrieval results with exact-match signals."""

    mode: str = "heuristic"
    top_n: int = 8

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Boost exact phrase, matched-term density, and title hits on the top candidate window."""

        if self.mode == "disabled":
            return sort_by_score(results)

        normalized_query = query.strip().lower()
        reranked: list[RetrievalResult] = []
        for index, result in enumerate(sort_by_score(results)):
            if index >= self.top_n:
                reranked.append(result)
                continue

            exact_phrase_boost = 0.12 if normalized_query and normalized_query in result.chunk.text.lower() else 0.0
            title_term_boost = 0.08 if any(term in result.document.title.lower() for term in result.matched_terms) else 0.0
            match_density_boost = min(len(result.matched_terms) * 0.02, 0.12)
            result.score = round(result.score + exact_phrase_boost + title_term_boost + match_density_boost, 4)
            reranked.append(result)
        return sort_by_score(reranked)
