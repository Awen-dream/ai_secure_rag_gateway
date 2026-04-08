from __future__ import annotations

from typing import Any, Sequence

import httpx


class OpenAIEmbeddingClient:
    """Thin embeddings API adapter with safe fallback behavior for local development."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout_seconds: float,
        dimensions: int | None = None,
        enabled: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.dimensions = dimensions
        self.enabled = enabled

    def can_execute(self) -> bool:
        """Return whether the client is configured to call a real embeddings API."""

        return self.enabled and bool(self.api_key)

    def build_payload(self, texts: Sequence[str]) -> dict[str, Any]:
        """Build one embeddings API request body for the provided texts."""

        payload: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Call the embeddings API and return vectors in input order."""

        if not self.can_execute():
            raise RuntimeError("Embedding client is not configured")

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self.build_payload(texts),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self.parse_embeddings(response.json())

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding for one text input."""

        return self.embed_texts([text])[0]

    @staticmethod
    def parse_embeddings(payload: dict[str, Any]) -> list[list[float]]:
        """Extract embedding vectors from a standard OpenAI-compatible API response."""

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("Embedding API returned no embedding data")

        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Embedding API returned an invalid embedding payload")
            vectors.append([float(value) for value in embedding])
        return vectors
