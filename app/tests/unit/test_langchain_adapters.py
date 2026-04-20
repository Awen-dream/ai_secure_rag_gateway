from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.infrastructure.frameworks.langchain_embeddings import LangChainEmbeddingAdapter
from app.infrastructure.frameworks.langchain_llm import LangChainChatClientAdapter, LangChainOpenAIResponsesAdapter


class _FakeChatOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def invoke(self, messages):
        return SimpleNamespace(content="LangChain 输出")


class _FakeOpenAIEmbeddings:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.3, 0.4]


class LangChainAdapterTest(unittest.TestCase):
    def test_langchain_chat_adapter_generates_text_when_runtime_available(self) -> None:
        with patch(
            "app.infrastructure.frameworks.langchain_llm.import_module",
            return_value=SimpleNamespace(ChatOpenAI=_FakeChatOpenAI),
        ):
            client = LangChainChatClientAdapter(
                provider="qwen",
                api_key="test-key",
                model="qwen-plus",
                base_url="https://example.com/v1",
                timeout_seconds=10,
                max_output_tokens=256,
                temperature=0.1,
            )

        self.assertTrue(client.can_execute())
        self.assertEqual(client.generate_response("system", "user"), "LangChain 输出")

    def test_langchain_embedding_adapter_embeds_texts_when_runtime_available(self) -> None:
        with patch(
            "app.infrastructure.frameworks.langchain_embeddings.import_module",
            return_value=SimpleNamespace(OpenAIEmbeddings=_FakeOpenAIEmbeddings),
        ):
            client = LangChainEmbeddingAdapter(
                api_key="test-key",
                model="text-embedding-3-small",
                base_url="https://example.com/v1",
                timeout_seconds=10,
                dimensions=256,
                enabled=True,
            )

        self.assertTrue(client.can_execute())
        self.assertEqual(client.embed_texts(["a", "b"]), [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(client.embed_text("a"), [0.3, 0.4])

    def test_langchain_runtime_selection_in_deps(self) -> None:
        from app.api.deps import (
            get_deepseek_client,
            get_embedding_client,
            get_openai_client,
            get_qwen_client,
        )

        original_llm_runtime = settings.llm_runtime
        original_embedding_runtime = settings.embedding_runtime
        original_embedding_provider = settings.embedding_provider
        original_openai_api_key = settings.openai_api_key
        original_qwen_api_key = settings.qwen_api_key
        original_deepseek_api_key = settings.deepseek_api_key
        original_embedding_api_key = settings.embedding_api_key
        try:
            settings.llm_runtime = "langchain"
            settings.embedding_runtime = "langchain"
            settings.embedding_provider = "openai"
            settings.openai_api_key = "openai-key"
            settings.qwen_api_key = "qwen-key"
            settings.deepseek_api_key = "deepseek-key"
            settings.embedding_api_key = "embedding-key"

            get_openai_client.cache_clear()
            get_embedding_client.cache_clear()
            with patch(
                "app.infrastructure.frameworks.langchain_llm.import_module",
                return_value=SimpleNamespace(ChatOpenAI=_FakeChatOpenAI),
            ), patch(
                "app.infrastructure.frameworks.langchain_embeddings.import_module",
                return_value=SimpleNamespace(OpenAIEmbeddings=_FakeOpenAIEmbeddings),
            ):
                openai_client = get_openai_client()
                qwen_client = get_qwen_client()
                deepseek_client = get_deepseek_client()
                embedding_client = get_embedding_client()

            self.assertIsInstance(openai_client, LangChainOpenAIResponsesAdapter)
            self.assertIsInstance(qwen_client, LangChainChatClientAdapter)
            self.assertIsInstance(deepseek_client, LangChainChatClientAdapter)
            self.assertIsInstance(embedding_client, LangChainEmbeddingAdapter)
        finally:
            settings.llm_runtime = original_llm_runtime
            settings.embedding_runtime = original_embedding_runtime
            settings.embedding_provider = original_embedding_provider
            settings.openai_api_key = original_openai_api_key
            settings.qwen_api_key = original_qwen_api_key
            settings.deepseek_api_key = original_deepseek_api_key
            settings.embedding_api_key = original_embedding_api_key
            get_openai_client.cache_clear()
            get_embedding_client.cache_clear()
