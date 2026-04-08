import unittest
from unittest.mock import patch

from app.infrastructure.llm.openai_embeddings import OpenAIEmbeddingClient


class OpenAIEmbeddingClientTest(unittest.TestCase):
    def test_build_payload_supports_dimensions(self) -> None:
        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            dimensions=512,
            enabled=True,
        )

        payload = client.build_payload(["alpha", "beta"])

        self.assertEqual(payload["model"], "text-embedding-3-small")
        self.assertEqual(payload["input"], ["alpha", "beta"])
        self.assertEqual(payload["dimensions"], 512)

    def test_parse_embeddings_extracts_vectors(self) -> None:
        vectors = OpenAIEmbeddingClient.parse_embeddings(
            {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }
        )

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])

    def test_embed_text_returns_first_vector_from_single_input(self) -> None:
        client = OpenAIEmbeddingClient(
            api_key="test-key",
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            timeout_seconds=30,
            enabled=True,
        )

        with patch.object(client, "embed_texts", return_value=[[0.5, 0.6]]) as embed_texts:
            vector = client.embed_text("hello world")

        embed_texts.assert_called_once_with(["hello world"])
        self.assertEqual(vector, [0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
