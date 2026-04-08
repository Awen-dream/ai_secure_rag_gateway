from __future__ import annotations

from app.application.query.understanding import QueryUnderstandingService
from app.application.query.retrieval_cache import RetrievalCache
from app.application.query.rewrite import rewrite_query
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
        query_understanding: QueryUnderstandingService | None = None,
    ) -> None:
        self.document_service = document_service
        self.keyword_backend = keyword_backend
        self.vector_backend = vector_backend
        self.retrieval_cache = retrieval_cache
        self.reranker = reranker
        self.query_understanding = query_understanding or QueryUnderstandingService()

    def retrieve(self, user: UserContext, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Run permission-aware hybrid retrieval and return fused evidence chunks."""

        explanation = self.explain(user, query, top_k)
        return explanation.results[: min(top_k, explanation.profile.top_k)]

    def explain(self, user: UserContext, query: str, top_k: int = 5) -> RetrievalExplainResponse:
        """Return an admin-friendly explanation of how hybrid retrieval handled the query."""

        understanding = self.query_understanding.understand(query)
        rewritten = understanding.rewritten_query or rewrite_query(query)
        profile = get_retrieval_profile(understanding.intent)
        terms = normalize_terms(rewritten)
        cached_results = None
        if self.retrieval_cache:
            cached_results = self.retrieval_cache.get_results(user, rewritten, min(top_k, profile.top_k))
        if cached_results is not None:
            results = cached_results
        else:
            results = self._hybrid_retrieve(user, rewritten, terms, profile)
            if self.retrieval_cache:
                self.retrieval_cache.set_results(user, rewritten, min(top_k, profile.top_k), results)
        return RetrievalExplainResponse(
            rewritten_query=rewritten,
            intent=understanding.intent,
            intent_confidence=understanding.confidence,
            intent_reasons=understanding.reasons,
            profile=profile,
            results=sort_by_score(results)[: min(top_k, profile.top_k)],
        )

    def backend_info(self) -> list[RetrievalBackendInfo]:
        """Return metadata describing the active keyword and vector retrieval backends."""

        return [
            self.keyword_backend.describe_backend(),
            self.vector_backend.describe_backend(),
        ]

    def _hybrid_retrieve(
        self,
        user: UserContext,
        query: str,
        terms: list[str],
        profile: RetrievalProfile,
    ) -> list[RetrievalResult]:
        """Fuse Elasticsearch-style keyword hits and PGVector-style semantic hits."""

        access_filter = build_access_filter(user)
        candidates = self.document_service.get_accessible_chunks(user)
        keyword_hits = self.keyword_backend.search(
            query=query,
            terms=terms,
            candidates=candidates,
            top_k=profile.candidate_pool,
            access_filter=access_filter,
        )
        vector_hits = self.vector_backend.search(
            query=query,
            candidates=candidates,
            top_k=profile.candidate_pool,
            access_filter=access_filter,
        )
        ranked_candidates = sort_by_score(self._merge_backend_hits(keyword_hits, vector_hits, profile, terms))
        if not ranked_candidates:
            return []

        top_score = ranked_candidates[0].score
        filtered_candidates = [
            item
            for item in ranked_candidates
            if item.score >= profile.min_score and item.score >= top_score * profile.relative_score_cutoff
        ]
        if self.reranker:
            filtered_candidates = self.reranker.rerank(query, filtered_candidates)
        return filtered_candidates[: profile.candidate_pool]

    @staticmethod
    def _merge_backend_hits(
        keyword_hits: list[BackendSearchHit],
        vector_hits: list[BackendSearchHit],
        profile: RetrievalProfile,
        terms: list[str],
    ) -> list[RetrievalResult]:
        """Normalize backend scores and fuse them into a single candidate set."""

        merged: dict[str, RetrievalResult] = {}
        max_keyword = max((hit.score for hit in keyword_hits), default=0.0)
        max_vector = max((hit.score for hit in vector_hits), default=0.0)

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
            title_boost = profile.title_boost if any(term in result.document.title.lower() for term in terms) else 0.0
            result.score = weighted_fusion(
                keyword_score=keyword_normalized,
                vector_score=vector_normalized,
                keyword_weight=profile.keyword_weight,
                vector_weight=profile.vector_weight,
                title_boost=title_boost,
            )
            result.keyword_score = round(keyword_normalized, 4)
            result.vector_score = round(vector_normalized, 4)
            normalized_results.append(result)
        return normalized_results
