import unittest

from app.infrastructure.llm.deepseek_client import DeepSeekClient
from app.infrastructure.llm.openai_client import OpenAIClient
from app.infrastructure.llm.qwen_client import QwenClient
from app.infrastructure.llm.router import LLMRouter


class LLMRouterTest(unittest.TestCase):
    def test_router_routes_each_purpose_to_configured_provider(self) -> None:
        router = LLMRouter.build(
            default_provider="openai",
            generation_provider="qwen",
            query_understanding_provider="deepseek",
            reranker_provider="openai",
            clients=[
                OpenAIClient(
                    api_key="openai-key",
                    model="gpt-5.4-mini",
                    base_url="https://api.openai.com/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
                QwenClient(
                    api_key="qwen-key",
                    model="qwen-plus",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
                DeepSeekClient(
                    api_key="deepseek-key",
                    model="deepseek-chat",
                    base_url="https://api.deepseek.com/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            ],
        )

        self.assertEqual(router.get_client("generation").provider, "qwen")
        self.assertEqual(router.get_client("query_understanding").provider, "deepseek")
        self.assertEqual(router.get_client("reranker").provider, "openai")

    def test_router_falls_back_to_default_when_requested_provider_is_unavailable(self) -> None:
        router = LLMRouter.build(
            default_provider="openai",
            generation_provider="qwen",
            query_understanding_provider="deepseek",
            reranker_provider="qwen",
            clients=[
                OpenAIClient(
                    api_key="openai-key",
                    model="gpt-5.4-mini",
                    base_url="https://api.openai.com/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
                QwenClient(
                    api_key=None,
                    model="qwen-plus",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
                DeepSeekClient(
                    api_key=None,
                    model="deepseek-chat",
                    base_url="https://api.deepseek.com/v1",
                    timeout_seconds=30,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            ],
        )

        self.assertEqual(router.get_client("generation").provider, "openai")
        self.assertEqual(router.get_client("query_understanding").provider, "openai")
        self.assertEqual(router.get_client("reranker").provider, "openai")


if __name__ == "__main__":
    unittest.main()
