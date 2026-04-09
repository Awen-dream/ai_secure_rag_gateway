import unittest

from app.application.query.rewrite import build_query_rewrite_plan, rewrite_query


class QueryRewritePlanTest(unittest.TestCase):
    def test_build_query_rewrite_plan_extracts_filters_and_expands_terms(self) -> None:
        plan = build_query_rewrite_plan("请问 2025年 #finance 最新报销审批时限是多少？")

        self.assertEqual(plan.tag_filters, ["finance"])
        self.assertEqual(plan.year_filters, [2025])
        self.assertTrue(plan.recency_hint)
        self.assertIn("报销", plan.expanded_terms)
        self.assertIn("费用报销", plan.expanded_terms)
        self.assertIn("审批时限", plan.expanded_terms)

    def test_rewrite_query_removes_polite_prefix_but_keeps_question_mark(self) -> None:
        rewritten = rewrite_query("请问 报销制度文档编号是多少？")

        self.assertEqual(rewritten, "报销制度文档编号是多少？")


if __name__ == "__main__":
    unittest.main()
