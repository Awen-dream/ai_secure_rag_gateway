import unittest

from app.application.query.planning import QueryPlanningService
from app.application.query.understanding import QueryUnderstandingResult


class _UnderstandingStub:
    def __init__(self, rewritten_query: str) -> None:
        self.rewritten_query = rewritten_query
        self.calls: list[dict] = []

    def understand(self, query: str, last_user_query=None, session_summary=None):
        self.calls.append(
            {
                "query": query,
                "last_user_query": last_user_query,
                "session_summary": session_summary,
            }
        )
        return QueryUnderstandingResult(
            rewritten_query=self.rewritten_query,
            intent="standard_qa",
            confidence=0.9,
            reasons=["stub"],
            source="rule",
            rule_rewritten_query=self.rewritten_query,
            rule_intent="standard_qa",
            rule_confidence=0.9,
            rule_reasons=["stub"],
        )


class QueryPlanningServiceTest(unittest.TestCase):
    def test_plan_combines_understanding_with_original_filters(self) -> None:
        service = QueryPlanningService(_UnderstandingStub("报销制度审批时限"))

        result = service.plan("请问 2025年 #finance 审批时限呢？", last_user_query="报销制度是什么？")

        self.assertEqual(result.rewritten_query, "报销制度审批时限")
        self.assertEqual(result.rewrite_plan.tag_filters, ["finance"])
        self.assertEqual(result.rewrite_plan.year_filters, [2025])
        self.assertIn("审批时限", result.rewrite_plan.expanded_terms)


if __name__ == "__main__":
    unittest.main()
