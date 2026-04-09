import unittest

from app.application.query.planning import QueryPlanningResult
from app.application.query.rewrite import build_query_rewrite_plan, refine_query_rewrite_plan
from app.application.query.understanding import QueryUnderstandingResult
from app.application.retrieval.planning import RecallPlanningService


class RecallPlanningServiceTest(unittest.TestCase):
    def test_plan_derives_profile_queries_and_filters(self) -> None:
        understanding = QueryUnderstandingResult(
            rewritten_query="报销制度审批时限",
            intent="standard_qa",
            confidence=0.9,
            reasons=["precomputed"],
            source="rule",
            rule_rewritten_query="报销制度审批时限",
            rule_intent="standard_qa",
            rule_confidence=0.9,
            rule_reasons=["precomputed"],
        )
        rewrite_plan = refine_query_rewrite_plan(
            build_query_rewrite_plan("请问 2025年 #finance 最新审批时限呢？", last_user_query="报销制度是什么？"),
            understanding.rewritten_query,
        )

        plan = RecallPlanningService().plan(
            QueryPlanningResult(
                understanding=understanding,
                rewrite_plan=rewrite_plan,
            ),
            top_k=9,
        )

        self.assertEqual(plan.profile.name, "standard_qa")
        self.assertEqual(plan.keyword_query, "报销制度审批时限")
        self.assertIn("审批时限", plan.keyword_terms)
        self.assertEqual(plan.filters.tag_filters, ["finance"])
        self.assertEqual(plan.filters.year_filters, [2025])
        self.assertIn("报销制度审批时限", plan.vector_query)
        self.assertEqual(plan.result_limit, 5)
        self.assertEqual(plan.candidate_pool, plan.profile.candidate_pool)
        self.assertIn("finance", plan.cache_key)


if __name__ == "__main__":
    unittest.main()
