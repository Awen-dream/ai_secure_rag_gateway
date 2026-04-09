from __future__ import annotations

from app.application.access.service import build_access_filter
from app.application.retrieval.rerank import RetrievalRerankService
from app.application.retrieval.planning import RecallFilterPlan, RecallPlan, RecallPlanningService
from app.application.query.planning import QueryPlanningResult, QueryPlanningService
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.backends import KeywordSearchBackend, VectorSearchBackend
from app.domain.retrieval.models import RetrievalBackendInfo, RetrievalExplainResponse, RetrievalResult
from app.domain.retrieval.rerankers import RetrievalReranker, sort_by_score


class RetrievalService:
    """Coordinates planning, backend execution, and rerank-layer selection for hybrid retrieval."""

    def __init__(
        self,
        document_service: DocumentService,
        keyword_backend: KeywordSearchBackend,
        vector_backend: VectorSearchBackend,
        retrieval_cache: RetrievalCache | None = None,
        reranker: RetrievalReranker | None = None,
        query_planning: QueryPlanningService | None = None,
        recall_planning: RecallPlanningService | None = None,
        rerank_service: RetrievalRerankService | None = None,
    ) -> None:
        self.document_service = document_service
        self.keyword_backend = keyword_backend
        self.vector_backend = vector_backend
        self.retrieval_cache = retrieval_cache
        self.query_planning = query_planning or QueryPlanningService()
        self.recall_planning = recall_planning or RecallPlanningService()
        self.rerank_service = rerank_service or RetrievalRerankService(reranker)

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
        recall_plan = self.recall_planning.plan(query_plan, top_k=top_k)
        understanding = query_plan.understanding
        rewrite_plan = query_plan.rewrite_plan
        cached_results = None
        if self.retrieval_cache:
            cached_results = self.retrieval_cache.get_results(user, recall_plan.cache_key, recall_plan.result_limit)
        if cached_results is not None:
            results = cached_results
            pre_rerank_results = []
        else:
            pre_rerank_results, results = self._hybrid_retrieve(user, recall_plan)
            if self.retrieval_cache:
                self.retrieval_cache.set_results(user, recall_plan.cache_key, recall_plan.result_limit, results)
        return RetrievalExplainResponse(
            rewritten_query=rewrite_plan.rewritten_query,
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
            rerank_sources=list(dict.fromkeys(result.rerank_source for result in results if result.rerank_source)),
            rerank_notes=list(
                dict.fromkeys(note for result in results for note in result.rerank_notes if note)
            )[:5],
            drop_reasons=list(
                dict.fromkeys(
                    reason
                    for result in pre_rerank_results
                    if result.selection_status == "dropped"
                    for reason in result.selection_reasons
                )
            )[:8],
            profile=recall_plan.profile,
            pre_rerank_results=sort_by_score(pre_rerank_results)[: min(recall_plan.candidate_pool, 10)],
            results=sort_by_score(results)[: recall_plan.result_limit],
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
        recall_plan: RecallPlan,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Execute backend retrieval and delegate candidate building/rerank to the rerank layer."""

        access_filter = build_access_filter(user)
        candidates = self._apply_query_filters(self.document_service.get_accessible_chunks(user), recall_plan.filters)
        if not candidates:
            return [], []
        keyword_hits = self.keyword_backend.search(
            query=recall_plan.keyword_query,
            terms=recall_plan.keyword_terms,
            candidates=candidates,
            top_k=recall_plan.candidate_pool,
            access_filter=access_filter,
            tag_filters=recall_plan.filters.tag_filters,
            year_filters=recall_plan.filters.year_filters,
            exact_terms=recall_plan.exact_match_terms,
        )
        vector_hits = self.vector_backend.search(
            query=recall_plan.vector_query,
            candidates=candidates,
            top_k=recall_plan.candidate_pool,
            access_filter=access_filter,
            tag_filters=recall_plan.filters.tag_filters,
            year_filters=recall_plan.filters.year_filters,
        )
        rerank_candidates = self.rerank_service.build_rerank_candidates(keyword_hits, vector_hits, recall_plan)
        rerank_execution = self.rerank_service.execute_rerank(rerank_candidates, recall_plan)
        return rerank_execution.pre_rerank_results, rerank_execution.results

    @staticmethod
    def _apply_query_filters(
        candidates: list[tuple],
        filters: RecallFilterPlan,
    ) -> list[tuple]:
        filtered = list(candidates)
        if filters.tag_filters:
            filtered = [
                item
                for item in filtered
                if any(tag.lower() in {doc_tag.lower() for doc_tag in item[0].tags} for tag in filters.tag_filters)
            ]
        if filters.year_filters:
            filtered = [
                item
                for item in filtered
                if item[0].updated_at.year in filters.year_filters or item[0].created_at.year in filters.year_filters
            ]
        return filtered
