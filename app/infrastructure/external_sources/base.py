from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExternalDocumentContent:
    """Normalized external-source payload consumed by the document ingestion pipeline."""

    title: str
    content: str
    source_type: str
    source_uri: str
    connector: str
    external_document_id: str
    external_version: str | None = None


class ExternalSourceConnector(Protocol):
    """Common connector contract used by source-specific sync services."""

    provider: str

    def parse_source(self, source: str):
        """Parse one source locator into a connector-specific reference."""

    def fetch_document(self, source: str) -> ExternalDocumentContent:
        """Fetch one external document and return a normalized payload."""

    def health_check(self) -> dict:
        """Return connector health for admin diagnostics."""
