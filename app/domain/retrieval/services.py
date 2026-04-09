from __future__ import annotations

from datetime import datetime

from app.application.query.planning import QueryPlanningResult, QueryPlanningService
from app.application.query.retrieval_cache import RetrievalCache
from app.application.query.rewrite import QueryRewritePlan
from app.domain.auth.filter_builder import build_access_filter
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.backends import BackendSearchHit, KeywordSearchBackend, VectorSearchBackend
from app.domain.retrieval.models import (
    RetrievalBackendInfo,
    RetrievalExplainResponse,
    RetrievalProfile,
    RetrievalResult,
)
from app.domain.retrieval.profiles import get_retrieval_profile
from app.domain.retrieval.rerankers import HeuristicReranker, sort_by_score, weighted_fusion
from app.domain.retrieval.retrievers import normalize_terms


class RetrievalService:
    """Coordinates hybrid retrieval across explicit keyword and vector backends."""

    def __init__(
        self,
        document_service: DocumentService,
        keyword_backend: KeywordSearchBackend,
        vector_backend: VectorSearchBackend,
        retrieval_cache: RetrievalCache | None = None,
        reranker: HeuristicReranker | None = None,
        query_planning: QueryPlanningService | None = None,
    ) -> None:
        self.document_service = document_service
        self.keyword_backend = keyword_backend
        self.vector_backend = vector_backend
        self.retrieval_cache = retrieval_cache
        self.reranker = reranker
        self.query_planning = query_planning or QueryPlanningService()

    def retrieve(
        self,
        user: UserContext,
        query: str,
        top_k: int = 5,
        query_plan: QueryPlanningResult | None = None,
    ) -> list[RetrievalResult]:
        """Run permission-aware hybrid retrieval and return fused evidence chunks."""

        explanation = self.explain(
            user,
            query,
            top_k,
            query_plan=query_plan,
        )
        return explanation.results[: min(top_k, explanation.profile.top_k)]

    def explain(
        self,
        user: UserContext,
        query: str,
        top_k: int = 5,
        query_plan: QueryPlanningResult | None = None,
    ) -> RetrievalExplainResponse:
        """Return an admin-friendly explanation of how hybrid retrieval handled the query."""

        query_plan = self._resolve_query_plan(
            query,
            query_plan=query_plan,
        )
        understanding = query_plan.understanding
        rewrite_plan = query_plan.rewrite_plan
        rewritten = rewrite_plan.rewritten_query
        profile = get_retrieval_profile(understanding.intent)
        cache_query = self._build_cache_query(rewrite_plan)
        cached_results = None
        if self.retrieval_cache:
            cached_results = self.retrieval_cache.get_results(user, cache_query, min(top_k, profile.top_k))
        if cached_results is not None:
            results = cached_results
        else:
            results = self._hybrid_retrieve(user, rewrite_plan, profile)
            if self.retrieval_cache:
                self.retrieval_cache.set_results(user, cache_query, min(top_k, profile.top_k), results)
        return RetrievalExplainResponse(
            rewritten_query=rewritten,
            intent=understanding.intent,
            intent_confidence=understanding.confidence,
            intent_reasons=understanding.reasons,
            understanding_source=understanding.source,
            rule_rewritten_query=understanding.rule_rewritten_query,
            rule_intent=understanding.rule_intent,
            rule_intent_confidence=understanding.rule_confidence,
            rule_intent_reasons=understanding.rule_reasons,
            query_keywords=rewrite_plan.keywords,
            expanded_terms=rewrite_plan.expanded_terms,
            tag_filters=rewrite_plan.tag_filters,
            year_filters=rewrite_plan.year_filters,
            recency_hint=rewrite_plan.recency_hint,
            profile=profile,
            results=sort_by_score(results)[: min(top_k, profile.top_k)],
        )

    def _resolve_query_plan(
        self,
        query: str,
        query_plan: QueryPlanningResult | None = None,
    ) -> QueryPlanningResult:
        if query_plan is not None:
            return query_plan
        return self.query_planning.plan(query)

    def backend_info(self) -> list[RetrievalBackendInfo]:
        """Return metadata describing the active keyword and vector retrieval backends."""

        return [
            self.keyword_backend.describe_backend(),
            self.vector_backend.describe_backend(),
        ]

    def _hybrid_retrieve(
        self,
        user: UserContext,
        rewrite_plan: QueryRewritePlan,
        profile: RetrievalProfile,
    ) -> list[RetrievalResult]:
        """Fuse Elasticsearch-style keyword hits and PGVector-style semantic hits."""

        access_filter = build_access_filter(user)
        candidates = self._apply_query_filters(self.document_service.get_accessible_chunks(user), rewrite_plan)
        if not candidates:
            return []
        keyword_terms = list(dict.fromkeys(rewrite_plan.expanded_terms or normalize_terms(rewrite_plan.rewritten_query)))
        keyword_hits = self.keyword_backend.search(
            query=rewrite_plan.rewritten_query,
            terms=keyword_terms,
            candidates=candidates,
            top_k=profile.candidate_pool,
            access_filter=access_filter,
        )
        vector_hits = self.vector_backend.search(
            query=self._build_vector_query(rewrite_plan),
            candidates=candidates,
            top_k=profile.candidate_pool,
            access_filter=access_filter,
        )
        ranked_candidates = sort_by_score(self._merge_backend_hits(keyword_hits, vector_hits, profile, rewrite_plan))
        if not ranked_candidates:
            return []

        top_score = ranked_candidates[0].score
        filtered_candidates = [
            item
            for item in ranked_candidates
            if item.score >= profile.min_score and item.score >= top_score * profile.relative_score_cutoff
        ]
        if self.reranker:
            filtered_candidates = self.reranker.rerank(rewrite_plan.rewritten_query, filtered_candidates)
        return filtered_candidates[: profile.candidate_pool]

    @staticmethod
    def _merge_backend_hits(
        keyword_hits: list[BackendSearchHit],
        vector_hits: list[BackendSearchHit],
        profile: RetrievalProfile,
        rewrite_plan: QueryRewritePlan,
    ) -> list[RetrievalResult]:
        """Normalize backend scores and fuse them into a single candidate set."""

        merged: dict[str, RetrievalResult] = {}
        max_keyword = max((hit.score for hit in keyword_hits), default=0.0)
        max_vector = max((hit.score for hit in vector_hits), default=0.0)
        keyword_rank = {hit.chunk.id: index + 1 for index, hit in enumerate(sorted(keyword_hits, key=lambda item: item.score, reverse=True))}
        vector_rank = {hit.chunk.id: index + 1 for index, hit in enumerate(sorted(vector_hits, key=lambda item: item.score, reverse=True))}

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
            phrase_boost = sum(0.04 for phrase in rewrite_plan.exact_phrases[:2] if phrase.lower() in result.chunk.text.lower())
            tag_boost = 0.05 if rewrite_plan.tag_filters and any(tag.lower() in result.document.tags for tag in rewrite_plan.tag_filters) else 0.0
            recency_boost = RetrievalService._recency_boost(result.document.updated_at, rewrite_plan.recency_hint)
            rank_fusion = RetrievalService._reciprocal_rank_fusion(
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
        return normalized_results

    @staticmethod
    def _build_cache_query(rewrite_plan: QueryRewritePlan) -> str:
        return "||".join(
            [
                rewrite_plan.rewritten_query,
                ",".join(rewrite_plan.tag_filters),
                ",".join(str(item) for item in rewrite_plan.year_filters),
                "recent" if rewrite_plan.recency_hint else "stable",
            ]
        )

    @staticmethod
    def _build_vector_query(rewrite_plan: QueryRewritePlan) -> str:
        terms = [rewrite_plan.rewritten_query]
        terms.extend(rewrite_plan.exact_phrases[:2])
        terms.extend(rewrite_plan.keywords[:6])
        return " ".join(dict.fromkeys(term for term in terms if term))

    @staticmethod
    def _apply_query_filters(
        candidates: list[tuple],
        rewrite_plan: QueryRewritePlan,
    ) -> list[tuple]:
        filtered = list(candidates)
        if rewrite_plan.tag_filters:
            filtered = [
                item
                for item in filtered
                if any(tag.lower() in {doc_tag.lower() for doc_tag in item[0].tags} for tag in rewrite_plan.tag_filters)
            ]
        if rewrite_plan.year_filters:
            filtered = [
                item
                for item in filtered
                if item[0].updated_at.year in rewrite_plan.year_filters or item[0].created_at.year in rewrite_plan.year_filters
            ]
        return filtered

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
