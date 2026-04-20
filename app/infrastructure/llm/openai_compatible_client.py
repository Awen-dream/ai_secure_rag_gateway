from __future__ import annotations

from typing import Any

import httpx


class OpenAICompatibleChatClient:
    """Call OpenAI-compatible `/chat/completions` providers and normalize text output."""

    provider = "openai-compatible"

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
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def can_execute(self) -> bool:
        """Return whether the client is configured to call a remote provider."""

        return bool(self.api_key)

    def build_payload(self, instructions: str, input_text: str) -> dict[str, Any]:
        """Build one OpenAI-compatible chat completions request body."""

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

    def generate_response(self, instructions: str, input_text: str) -> str:
        """Call the provider's chat completions endpoint and return normalized text output."""

        if not self.can_execute():
            raise RuntimeError(f"{self.provider} client is not configured")

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self.build_payload(instructions, input_text),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self.parse_output_text(response.json())

    def parse_output_text(self, payload: dict[str, Any]) -> str:
        """Extract normalized text from common OpenAI-compatible chat completion shapes."""

        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError(f"{self.provider} returned no choices")

        text_fragments: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                text_fragments.append(content.strip())
                continue
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                        fragment = item["text"].strip()
                        if fragment:
                            text_fragments.append(fragment)

        normalized = "\n".join(text_fragments).strip()
        if normalized:
            return normalized
        raise RuntimeError(f"{self.provider} returned no text output")
