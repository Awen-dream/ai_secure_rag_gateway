import unittest
from unittest.mock import Mock

from app.application.query.understanding import QueryUnderstandingService
from app.infrastructure.llm.openai_client import OpenAIClient


class QueryUnderstandingServiceTest(unittest.TestCase):
    def test_falls_back_to_rule_understanding_when_openai_is_unavailable(self) -> None:
        client = OpenAIClient(
            api_key=None,
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=256,
            temperature=0.0,
        )
        service = QueryUnderstandingService(client)

        result = service.understand("报销制度文档编号是多少？")

        self.assertEqual(result.intent, "exact_lookup")
        self.assertEqual(result.source, "rule")
        self.assertEqual(result.rewritten_query, "报销制度文档编号是多少？")
        self.assertEqual(result.rule_intent, "exact_lookup")
        self.assertEqual(result.rule_rewritten_query, "报销制度文档编号是多少？")

    def test_uses_llm_result_when_json_is_valid(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=256,
            temperature=0.0,
        )
        client.generate_response = Mock(
            return_value=(
                '{"rewritten_query":"报销制度和采购制度的审批差异",'
                '"intent":"summary","confidence":0.91,"reasons":["comparison_language"]}'
            )
        )
        service = QueryUnderstandingService(client)

        result = service.understand("对比报销和采购审批差异")

        self.assertEqual(result.intent, "summary")
        self.assertEqual(result.source, "llm")
        self.assertEqual(result.rewritten_query, "报销制度和采购制度的审批差异")
        self.assertIn("llm_query_understanding", result.reasons)
        self.assertEqual(result.rule_intent, "summary")

    def test_returns_fallback_when_llm_output_is_invalid(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=256,
            temperature=0.0,
        )
        client.generate_response = Mock(return_value="not-json")
        service = QueryUnderstandingService(client)

        result = service.understand("请总结一下报销制度")

        self.assertEqual(result.intent, "summary")
        self.assertEqual(result.source, "fallback")
        self.assertIn("llm_fallback", result.reasons)
        self.assertEqual(result.rule_intent, "summary")

    def test_rewrite_guardrail_preserves_critical_terms(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=256,
            temperature=0.0,
        )
        client.generate_response = Mock(
            return_value=(
                '{"rewritten_query":"查看报销制度",'
                '"intent":"exact_lookup","confidence":0.93,"reasons":["metadata_lookup_pattern"]}'
            )
        )
        service = QueryUnderstandingService(client)

        result = service.understand("报销制度第2版编号是多少？")

        self.assertEqual(result.rewritten_query, result.rule_rewritten_query)
        self.assertIn("missing_critical_terms", result.reasons)


if __name__ == "__main__":
    unittest.main()
