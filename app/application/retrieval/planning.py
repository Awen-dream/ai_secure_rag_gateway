from __future__ import annotations

from dataclasses import dataclass, field

from app.application.query.planning import QueryPlanningResult
from app.domain.retrieval.models import RetrievalProfile
from app.domain.retrieval.profiles import get_retrieval_profile
from app.domain.retrieval.retrievers import normalize_terms


@dataclass(frozen=True)
class RecallFilterPlan:
    tag_filters: list[str] = field(default_factory=list)
    year_filters: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RecallPlan:
    query_plan: QueryPlanningResult
    profile: RetrievalProfile
    keyword_query: str
    keyword_terms: list[str]
    vector_query: str
    filters: RecallFilterPlan
    cache_key: str
    result_limit: int
    candidate_pool: int


class RecallPlanningService:
    """Translate query planning output into a concrete retrieval plan without executing it."""

    def plan(self, query_plan: QueryPlanningResult, top_k: int = 5) -> RecallPlan:
        rewrite_plan = query_plan.rewrite_plan
        profile = get_retrieval_profile(query_plan.understanding.intent)
        keyword_terms = list(dict.fromkeys(rewrite_plan.expanded_terms or normalize_terms(rewrite_plan.rewritten_query)))
        vector_query = self._build_vector_query(rewrite_plan)
        filters = RecallFilterPlan(
            tag_filters=list(rewrite_plan.tag_filters),
            year_filters=list(rewrite_plan.year_filters),
        )
        return RecallPlan(
            query_plan=query_plan,
            profile=profile,
            keyword_query=rewrite_plan.rewritten_query,
            keyword_terms=keyword_terms,
            vector_query=vector_query,
            filters=filters,
            cache_key=self._build_cache_key(rewrite_plan),
            result_limit=min(top_k, profile.top_k),
            candidate_pool=profile.candidate_pool,
        )

    @staticmethod
    def _build_cache_key(rewrite_plan) -> str:
        return "||".join(
            [
                rewrite_plan.rewritten_query,
                ",".join(rewrite_plan.tag_filters),
                ",".join(str(item) for item in rewrite_plan.year_filters),
                "recent" if rewrite_plan.recency_hint else "stable",
            ]
        )

    @staticmethod
    def _build_vector_query(rewrite_plan) -> str:
        terms = [rewrite_plan.rewritten_query]
        terms.extend(rewrite_plan.exact_phrases[:2])
        terms.extend(rewrite_plan.keywords[:6])
        return " ".join(dict.fromkeys(term for term in terms if term))
