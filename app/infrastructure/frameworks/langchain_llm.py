from __future__ import annotations

from importlib import import_module
from typing import Any

from app.infrastructure.llm.openai_compatible_client import OpenAICompatibleChatClient
from app.infrastructure.llm.openai_client import OpenAIClient


class LangChainChatClientAdapter(OpenAICompatibleChatClient):
    """LangChain-backed chat client for OpenAI-compatible providers with native fallback semantics."""

    def __init__(
        self,
        provider: str,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self._chat_model = self._build_chat_model()

    def can_execute(self) -> bool:
        return bool(self.api_key) and self._chat_model is not None

    def generate_response(self, instructions: str, input_text: str) -> str:
        if not self.can_execute():
            raise RuntimeError(f"{self.provider} langchain client is not configured")

        messages = [
            ("system", instructions),
            ("human", input_text),
        ]
        response = self._chat_model.invoke(messages)
        normalized = self._extract_text(response)
        if normalized:
            return normalized
        raise RuntimeError(f"{self.provider} langchain client returned no text output")

    def _build_chat_model(self) -> Any | None:
        try:
            module = import_module("langchain_openai")
        except Exception:
            return None

        chat_class = getattr(module, "ChatOpenAI", None)
        if chat_class is None:
            return None

        try:
            return chat_class(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        except Exception:
            return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    fragments.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        fragments.append(text.strip())
            return "\n".join(fragments).strip()
        return ""


class LangChainOpenAIResponsesAdapter(OpenAIClient):
    """LangChain-backed OpenAI adapter that preserves the OpenAI provider name for router compatibility."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self._delegate = LangChainChatClientAdapter(
            provider=self.provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    def can_execute(self) -> bool:
        return self._delegate.can_execute()

    def generate_response(self, instructions: str, input_text: str) -> str:
        return self._delegate.generate_response(instructions, input_text)
