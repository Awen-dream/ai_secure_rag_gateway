from __future__ import annotations

from app.infrastructure.llm.openai_compatible_client import OpenAICompatibleChatClient


class DeepSeekClient(OpenAICompatibleChatClient):
    """DeepSeek chat client via its OpenAI-compatible chat completions API."""

    provider = "deepseek"

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
            provider=self.provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
