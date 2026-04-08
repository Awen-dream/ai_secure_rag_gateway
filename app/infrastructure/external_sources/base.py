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


@dataclass(frozen=True)
class ExternalSourceItem:
    """One externally discoverable source entry returned by a connector listing call."""

    source: str
    source_kind: str
    external_document_id: str
    title: str | None = None
    space_id: str | None = None
    node_token: str | None = None
    parent_node_token: str | None = None
    obj_type: str | None = None
    has_child: bool = False


@dataclass(frozen=True)
class ExternalSourcePage:
    """One paginated connector listing response."""

    items: list[ExternalSourceItem]
    next_cursor: str | None = None


class ExternalSourceConnector(Protocol):
    """Common connector contract used by source-specific sync services."""

    provider: str

    def parse_source(self, source: str):
        """Parse one source locator into a connector-specific reference."""

    def fetch_document(self, source: str) -> ExternalDocumentContent:
        """Fetch one external document and return a normalized payload."""

    def list_sources(
        self,
        cursor: str | None = None,
        limit: int = 20,
        source_root: str | None = None,
        space_id: str | None = None,
        parent_node_token: str | None = None,
    ) -> ExternalSourcePage:
        """List externally discoverable sources for incremental or paginated sync."""

    def health_check(self) -> dict:
        """Return connector health for admin diagnostics."""
