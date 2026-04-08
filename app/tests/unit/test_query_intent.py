import unittest

from app.application.query.intent import classify_query_intent, classify_query_intent_details


class QueryIntentClassificationTest(unittest.TestCase):
    def test_classifies_exact_lookup_for_document_identifier_queries(self) -> None:
        self.assertEqual(classify_query_intent("报销制度文档编号是多少？"), "exact_lookup")
        self.assertEqual(classify_query_intent("请帮我查一下报销制度第2版在哪里看"), "exact_lookup")

    def test_classifies_summary_for_summary_and_comparison_queries(self) -> None:
        self.assertEqual(classify_query_intent("请总结一下报销制度的审批规则"), "summary")
        self.assertEqual(classify_query_intent("对比采购制度和报销制度的审批差异"), "summary")
        self.assertEqual(classify_query_intent("报销制度 v2 和 v3 有什么区别"), "summary")

    def test_keeps_standard_qa_for_general_questions_with_incidental_id_tokens(self) -> None:
        self.assertEqual(classify_query_intent("系统里的 idempotency key 是什么？"), "standard_qa")
        self.assertEqual(classify_query_intent("报销审批时限是什么？"), "standard_qa")

    def test_returns_confidence_and_reason_codes_for_observability(self) -> None:
        result = classify_query_intent_details("请帮我查一下报销制度第2版在哪里看")

        self.assertEqual(result.intent, "exact_lookup")
        self.assertGreater(result.confidence, 0.8)
        self.assertIn("explicit_version_pattern", result.reasons)


if __name__ == "__main__":
    unittest.main()
