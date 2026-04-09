from __future__ import annotations

from dataclasses import dataclass

from app.application.query.rewrite import QueryRewritePlan, build_query_rewrite_plan, refine_query_rewrite_plan
from app.application.query.understanding import QueryUnderstandingResult, QueryUnderstandingService


@dataclass(frozen=True)
class QueryPlanningResult:
    understanding: QueryUnderstandingResult
    rewrite_plan: QueryRewritePlan

    @property
    def rewritten_query(self) -> str:
        return self.rewrite_plan.rewritten_query


class QueryPlanningService:
    """Compose query understanding and rewrite planning into one reusable planning step."""

    def __init__(self, query_understanding: QueryUnderstandingService | None = None) -> None:
        self.query_understanding = query_understanding or QueryUnderstandingService()

    def plan(
        self,
        query: str,
        last_user_query: str | None = None,
        session_summary: str | None = None,
        understanding: QueryUnderstandingResult | None = None,
    ) -> QueryPlanningResult:
        resolved_understanding = understanding or self.query_understanding.understand(
            query,
            last_user_query=last_user_query,
            session_summary=session_summary,
        )
        base_plan = build_query_rewrite_plan(
            query,
            last_user_query=last_user_query,
            session_summary=session_summary,
        )
        rewrite_plan = refine_query_rewrite_plan(base_plan, resolved_understanding.rewritten_query)
        return QueryPlanningResult(
            understanding=resolved_understanding,
            rewrite_plan=rewrite_plan,
        )
