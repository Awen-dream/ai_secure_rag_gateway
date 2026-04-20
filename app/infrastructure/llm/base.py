from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Common interface shared by generation-capable LLM providers."""

    provider: str
    model: str

    def can_execute(self) -> bool:
        """Return whether the client is configured to call its remote provider."""

    def generate_response(self, instructions: str, input_text: str) -> str:
        """Generate one plain-text response from instructions plus input text."""
