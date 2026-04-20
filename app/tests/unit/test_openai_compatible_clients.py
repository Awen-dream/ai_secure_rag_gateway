import unittest

from app.infrastructure.llm.deepseek_client import DeepSeekClient
from app.infrastructure.llm.qwen_client import QwenClient


class OpenAICompatibleClientTest(unittest.TestCase):
    def test_qwen_build_payload_matches_chat_completions_shape(self) -> None:
        client = QwenClient(
            api_key="test-key",
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        payload = client.build_payload("system instructions", "user input")

        self.assertEqual(payload["model"], "qwen-plus")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "system instructions")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "user input")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.2)

    def test_deepseek_parse_output_text_supports_string_content(self) -> None:
        client = DeepSeekClient(
            api_key="test-key",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        text = client.parse_output_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": "结论：审批时限为3个工作日。\n依据：[1] 报销制度。"
                        }
                    }
                ]
            }
        )

        self.assertEqual(text, "结论：审批时限为3个工作日。\n依据：[1] 报销制度。")

    def test_qwen_parse_output_text_supports_content_parts(self) -> None:
        client = QwenClient(
            api_key="test-key",
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout_seconds=30,
            max_output_tokens=512,
            temperature=0.2,
        )

        text = client.parse_output_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "结论：审批时限为3个工作日。"},
                                {"type": "text", "text": "依据：[1] 报销制度。"},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(text, "结论：审批时限为3个工作日。\n依据：[1] 报销制度。")


if __name__ == "__main__":
    unittest.main()
