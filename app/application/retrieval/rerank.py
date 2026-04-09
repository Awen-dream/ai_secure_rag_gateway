from __future__ import annotations

from datetime import datetime

from app.application.retrieval.planning import RecallPlan
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalProfile, RetrievalResult
from app.domain.retrieval.rerankers import HeuristicReranker, sort_by_score, weighted_fusion


class RetrievalRerankService:
    """Prepare rerank candidates from multi-backend hits and run the configured reranker."""

    def __init__(self, reranker: HeuristicReranker | None = None) -> None:
        self.reranker = reranker

    def build_rerank_candidates(
        self,
        keyword_hits: list[BackendSearchHit],
        vector_hits: list[BackendSearchHit],
        recall_plan: RecallPlan,
    ) -> list[RetrievalResult]:
        profile = recall_plan.profile
        rewrite_plan = recall_plan.query_plan.rewrite_plan
        merged: dict[str, RetrievalResult] = {}
        max_keyword = max((hit.score for hit in keyword_hits), default=0.0)
        max_vector = max((hit.score for hit in vector_hits), default=0.0)
        keyword_rank = {
            hit.chunk.id: index + 1
            for index, hit in enumerate(sorted(keyword_hits, key=lambda item: item.score, reverse=True))
        }
        vector_rank = {
            hit.chunk.id: index + 1
            for index, hit in enumerate(sorted(vector_hits, key=lambda item: item.score, reverse=True))
        }

        for hit in keyword_hits:
            merged[hit.chunk.id] = RetrievalResult(
                document=hit.document,
                chunk=hit.chunk,
                score=0.0,
                keyword_score=hit.score,
                vector_score=0.0,
                matched_terms=hit.matched_terms,
                retrieval_sources=[hit.backend],
            )

        for hit in vector_hits:
            if hit.chunk.id not in merged:
                merged[hit.chunk.id] = RetrievalResult(
                    document=hit.document,
                    chunk=hit.chunk,
                    score=0.0,
                    keyword_score=0.0,
                    vector_score=hit.score,
                    matched_terms=[],
                    retrieval_sources=[hit.backend],
                )
                continue

            existing = merged[hit.chunk.id]
            existing.vector_score = hit.score
            if hit.backend not in existing.retrieval_sources:
                existing.retrieval_sources.append(hit.backend)

        normalized_results: list[RetrievalResult] = []
        for result in merged.values():
            keyword_normalized = result.keyword_score / max_keyword if max_keyword else 0.0
            vector_normalized = result.vector_score / max_vector if max_vector else 0.0
            title_boost = (
                profile.title_boost
                if any(term in result.document.title.lower() for term in rewrite_plan.expanded_terms or rewrite_plan.keywords)
                else 0.0
            )
            phrase_boost = sum(
                0.04 for phrase in rewrite_plan.exact_phrases[:2] if phrase.lower() in result.chunk.text.lower()
            )
            tag_boost = (
                0.05
                if rewrite_plan.tag_filters and any(tag.lower() in result.document.tags for tag in rewrite_plan.tag_filters)
                else 0.0
            )
            recency_boost = self._recency_boost(result.document.updated_at, rewrite_plan.recency_hint)
            rank_fusion = self._reciprocal_rank_fusion(
                keyword_rank.get(result.chunk.id),
                vector_rank.get(result.chunk.id),
                profile,
            )
            result.score = weighted_fusion(
                keyword_score=keyword_normalized,
                vector_score=vector_normalized,
                keyword_weight=profile.keyword_weight,
                vector_weight=profile.vector_weight,
                title_boost=title_boost,
            )
            result.score = round(result.score + phrase_boost + tag_boost + recency_boost + rank_fusion, 4)
            result.keyword_score = round(keyword_normalized, 4)
            result.vector_score = round(vector_normalized, 4)
            normalized_results.append(result)
        return sort_by_score(normalized_results)

    def rerank_results(self, results: list[RetrievalResult], recall_plan: RecallPlan) -> list[RetrievalResult]:
        if not results:
            return []

        rewrite_plan = recall_plan.query_plan.rewrite_plan
        top_score = results[0].score
        filtered_candidates = [
            item
            for item in results
            if item.score >= recall_plan.profile.min_score
            and item.score >= top_score * recall_plan.profile.relative_score_cutoff
        ]
        if self.reranker:
            filtered_candidates = self.reranker.rerank(rewrite_plan.rewritten_query, filtered_candidates)
        return filtered_candidates[: recall_plan.candidate_pool]

    @staticmethod
    def _reciprocal_rank_fusion(
        keyword_rank: int | None,
        vector_rank: int | None,
        profile: RetrievalProfile,
        k: int = 60,
    ) -> float:
        score = 0.0
        if keyword_rank is not None:
            score += (1.0 / (k + keyword_rank)) * (0.45 + profile.keyword_weight)
        if vector_rank is not None:
            score += (1.0 / (k + vector_rank)) * (0.45 + profile.vector_weight)
        return round(score, 4)

    @staticmethod
    def _recency_boost(updated_at: datetime, recency_hint: bool) -> float:
        if not recency_hint:
            return 0.0
        age_days = max((datetime.utcnow() - updated_at).days, 0)
        if age_days <= 30:
            return 0.06
        if age_days <= 180:
            return 0.03
        if age_days <= 365:
            return 0.015
        return 0.0
