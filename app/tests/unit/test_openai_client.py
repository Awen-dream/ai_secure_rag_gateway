import unittest

from app.infrastructure.llm.openai_client import OpenAIClient


class OpenAIClientTest(unittest.TestCase):
    def test_can_execute_depends_on_api_key(self) -> None:
        disabled_client = OpenAIClient(
            api_key=None,
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=900,
            temperature=0.1,
        )
        enabled_client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=900,
            temperature=0.1,
        )

        self.assertFalse(disabled_client.can_execute())
        self.assertTrue(enabled_client.can_execute())

    def test_build_payload_matches_responses_api_shape(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        payload = client.build_payload("system instructions", "user input")

        self.assertEqual(payload["model"], "gpt-5.4-mini")
        self.assertEqual(payload["instructions"], "system instructions")
        self.assertEqual(payload["input"], "user input")
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.2)

    def test_parse_output_text_prefers_top_level_field(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        text = client.parse_output_text({"output_text": "结论：审批时限为3个工作日。"})

        self.assertEqual(text, "结论：审批时限为3个工作日。")

    def test_parse_output_text_supports_message_content_shape(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-5.4-mini",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        text = client.parse_output_text(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "结论：审批时限为3个工作日。"},
                            {"type": "output_text", "text": "依据：[1] 报销制度。"},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(text, "结论：审批时限为3个工作日。\n依据：[1] 报销制度。")


if __name__ == "__main__":
    unittest.main()
