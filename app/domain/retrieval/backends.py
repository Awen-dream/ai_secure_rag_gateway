from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, Field

from app.domain.auth.filter_builder import AccessFilter
from app.domain.documents.models import DocumentChunk, DocumentRecord


class BackendSearchHit(BaseModel):
    """Represents a raw backend hit before hybrid fusion is applied."""

    document: DocumentRecord
    chunk: DocumentChunk
    score: float
    backend: str
    matched_terms: list[str] = Field(default_factory=list)


class KeywordSearchBackend(Protocol):
    """Contract for keyword-oriented retrieval backends such as Elasticsearch."""

    backend_name: str

    def search(
        self,
        query: str,
        terms: Sequence[str],
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None = None,
    ) -> list[BackendSearchHit]:
        """Search keyword evidence from permission-filtered candidates."""

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Sync the latest document snapshot into the keyword index."""

    def delete_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Remove one document snapshot from the keyword index."""

    def describe_backend(self):
        """Return deployment metadata for this keyword backend."""


class VectorSearchBackend(Protocol):
    """Contract for vector-oriented retrieval backends such as PGVector."""

    backend_name: str

    def search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None = None,
    ) -> list[BackendSearchHit]:
        """Search semantic evidence from permission-filtered candidates."""

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Sync the latest document snapshot into the vector index."""

    def delete_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Remove one document snapshot from the vector index."""

    def describe_backend(self):
        """Return deployment metadata for this vector backend."""
