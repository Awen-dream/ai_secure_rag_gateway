from __future__ import annotations

from importlib import import_module
from typing import Any, Sequence

from app.infrastructure.llm.openai_embeddings import OpenAIEmbeddingClient


class LangChainEmbeddingAdapter(OpenAIEmbeddingClient):
    """LangChain-backed embedding adapter that preserves the current embedding client interface."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout_seconds: float,
        dimensions: int | None = None,
        enabled: bool = False,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            dimensions=dimensions,
            enabled=enabled,
        )
        self._embeddings = self._build_embeddings()

    def can_execute(self) -> bool:
        return bool(self.enabled and self.api_key and self._embeddings is not None)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.can_execute():
            raise RuntimeError("LangChain embedding adapter is not configured")
        vectors = self._embeddings.embed_documents(list(texts))
        return [[float(value) for value in vector] for vector in vectors]

    def embed_text(self, text: str) -> list[float]:
        if not self.can_execute():
            raise RuntimeError("LangChain embedding adapter is not configured")
        vector = self._embeddings.embed_query(text)
        return [float(value) for value in vector]

    def _build_embeddings(self) -> Any | None:
        try:
            module = import_module("langchain_openai")
        except Exception:
            return None

        embeddings_class = getattr(module, "OpenAIEmbeddings", None)
        if embeddings_class is None:
            return None

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,
            "base_url": self.base_url,
            "request_timeout": self.timeout_seconds,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        try:
            return embeddings_class(**kwargs)
        except Exception:
            return None
