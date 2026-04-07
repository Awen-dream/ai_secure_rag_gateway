from __future__ import annotations

from typing import Any

import httpx


class OpenAIClient:
    """Thin Responses API adapter used by the chat service for authorized RAG generation."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def can_execute(self) -> bool:
        """Return whether the client is configured to call the real OpenAI Responses API."""

        return bool(self.api_key)

    def build_payload(self, instructions: str, input_text: str) -> dict[str, Any]:
        """Build one Responses API request body from rendered prompt instructions and input."""

        return {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }

    def generate_response(self, instructions: str, input_text: str) -> str:
        """Call the OpenAI Responses API and return normalized plain text output."""

        if not self.can_execute():
            raise RuntimeError("OpenAI client is not configured")

        payload = self.build_payload(instructions, input_text)
        response = httpx.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self.parse_output_text(response.json())

    def parse_output_text(self, payload: dict[str, Any]) -> str:
        """Extract normalized text from the Responses API payload across common response shapes."""

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        text_fragments: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    text_fragments.append(content["text"].strip())

        normalized = "\n".join(fragment for fragment in text_fragments if fragment).strip()
        if normalized:
            return normalized
        raise RuntimeError("OpenAI Responses API returned no text output")
