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
    exact_match_terms: list[str]
    filters: RecallFilterPlan
    cache_key: str
    result_limit: int
    candidate_pool: int
    max_chunks_per_document: int


class RecallPlanningService:
    """Translate query planning output into a concrete retrieval plan without executing it."""

    def plan(self, query_plan: QueryPlanningResult, top_k: int = 5) -> RecallPlan:
        rewrite_plan = query_plan.rewrite_plan
        profile = self._resolve_profile(query_plan)
        keyword_terms = self._build_keyword_terms(rewrite_plan)
        exact_match_terms = self._build_exact_match_terms(rewrite_plan)
        vector_query = self._build_vector_query(rewrite_plan)
        filters = RecallFilterPlan(
            tag_filters=list(rewrite_plan.tag_filters),
            year_filters=list(rewrite_plan.year_filters),
        )
        candidate_pool = self._resolve_candidate_pool(profile, rewrite_plan)
        return RecallPlan(
            query_plan=query_plan,
            profile=profile,
            keyword_query=rewrite_plan.rewritten_query,
            keyword_terms=keyword_terms,
            vector_query=vector_query,
            exact_match_terms=exact_match_terms,
            filters=filters,
            cache_key=self._build_cache_key(rewrite_plan),
            result_limit=min(top_k, profile.top_k),
            candidate_pool=candidate_pool,
            max_chunks_per_document=1 if profile.name == "exact_lookup" else 2,
        )

    @staticmethod
    def _resolve_profile(query_plan: QueryPlanningResult) -> RetrievalProfile:
        rewrite_plan = query_plan.rewrite_plan
        understanding = query_plan.understanding
        rewritten = rewrite_plan.rewritten_query.strip()
        short_exact_query = bool(rewrite_plan.exact_phrases) and len(rewritten) <= 18 and not rewrite_plan.recency_hint
        if understanding.intent == "standard_qa" and short_exact_query:
            return get_retrieval_profile("exact_lookup")
        return get_retrieval_profile(understanding.intent)

    @staticmethod
    def _build_keyword_terms(rewrite_plan) -> list[str]:
        terms: list[str] = []
        terms.extend(phrase.lower() for phrase in rewrite_plan.exact_phrases if phrase.strip())
        terms.extend(rewrite_plan.expanded_terms)
        if not terms:
            terms.extend(normalize_terms(rewrite_plan.rewritten_query))
        return list(dict.fromkeys(term for term in terms if term))

    @staticmethod
    def _build_exact_match_terms(rewrite_plan) -> list[str]:
        terms: list[str] = []
        terms.extend(phrase.lower() for phrase in rewrite_plan.exact_phrases if phrase.strip())
        terms.extend(term.lower() for term in rewrite_plan.keywords if len(term) >= 4)
        return list(dict.fromkeys(terms))[:4]

    @staticmethod
    def _resolve_candidate_pool(profile: RetrievalProfile, rewrite_plan) -> int:
        bonus = 0
        if rewrite_plan.tag_filters:
            bonus += 2
        if rewrite_plan.year_filters or rewrite_plan.recency_hint:
            bonus += 2
        if rewrite_plan.exact_phrases:
            bonus += 1
        return profile.candidate_pool + bonus

    @staticmethod
    def _build_cache_key(rewrite_plan) -> str:
        return "||".join(
            [
                rewrite_plan.rewritten_query,
                ",".join(rewrite_plan.exact_phrases[:3]),
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
