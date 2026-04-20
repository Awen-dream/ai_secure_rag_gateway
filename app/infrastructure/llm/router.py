from __future__ import annotations

from dataclasses import dataclass

from app.infrastructure.llm.base import LLMClient

_SUPPORTED_PROVIDERS = ("openai", "qwen", "deepseek")
_SUPPORTED_PURPOSES = ("generation", "query_understanding", "reranker")


@dataclass(frozen=True)
class LLMRouter:
    """Select an LLM provider per purpose with graceful fallback to available clients."""

    default_provider: str
    purpose_providers: dict[str, str]
    clients: dict[str, LLMClient]

    def get_client(self, purpose: str) -> LLMClient:
        """Return the best client for the given purpose based on config and availability."""

        requested_provider = self._normalize_provider(self.purpose_providers.get(purpose, self.default_provider))
        fallback_order = [requested_provider, self.default_provider, *_SUPPORTED_PROVIDERS]
        seen: set[str] = set()
        first_known_client: LLMClient | None = None

        for provider in fallback_order:
            normalized = self._normalize_provider(provider)
            if normalized in seen:
                continue
            seen.add(normalized)
            client = self.clients.get(normalized)
            if not client:
                continue
            if first_known_client is None:
                first_known_client = client
            if client.can_execute():
                return client

        if first_known_client is not None:
            return first_known_client
        raise KeyError(f"No LLM clients are registered for purpose={purpose}")

    @staticmethod
    def build(
        default_provider: str,
        generation_provider: str,
        query_understanding_provider: str,
        reranker_provider: str,
        clients: list[LLMClient],
    ) -> "LLMRouter":
        """Construct one router instance from configured providers and client objects."""

        purpose_providers = {
            "generation": generation_provider,
            "query_understanding": query_understanding_provider,
            "reranker": reranker_provider,
        }
        return LLMRouter(
            default_provider=LLMRouter._normalize_provider(default_provider),
            purpose_providers={
                purpose: LLMRouter._normalize_provider(provider)
                for purpose, provider in purpose_providers.items()
                if purpose in _SUPPORTED_PURPOSES
            },
            clients={client.provider: client for client in clients},
        )

    @staticmethod
    def _normalize_provider(provider: str | None) -> str:
        normalized = (provider or "openai").strip().lower()
        return normalized if normalized in _SUPPORTED_PROVIDERS else "openai"
