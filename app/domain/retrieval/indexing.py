from __future__ import annotations

from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import KeywordSearchBackend, VectorSearchBackend


class RetrievalIndexingService:
    """Coordinates index updates across keyword and vector retrieval backends."""

    def __init__(
        self,
        keyword_backend: KeywordSearchBackend,
        vector_backend: VectorSearchBackend,
    ) -> None:
        self.keyword_backend = keyword_backend
        self.vector_backend = vector_backend

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Push a document snapshot to both retrieval backends after ingestion or reindex."""

        return {
            "keyword": self.keyword_backend.upsert_document(document, chunks),
            "vector": self.vector_backend.upsert_document(document, chunks),
        }
