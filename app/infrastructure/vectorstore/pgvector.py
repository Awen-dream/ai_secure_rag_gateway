from __future__ import annotations

from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.retrievers import vector_score


class PGVectorStore:
    """Development-safe PGVector adapter.

    The adapter exposes the same methods we expect from a real PostgreSQL + pgvector
    implementation. For now it computes semantic similarity locally, which keeps the
    hybrid retrieval flow runnable in tests and local development.
    """

    backend_name = "pgvector"

    def __init__(self, table_name: str = "document_embeddings") -> None:
        self.table_name = table_name

    def search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
    ) -> list[BackendSearchHit]:
        """Return semantic hits scored with a lightweight local similarity fallback."""

        hits: list[BackendSearchHit] = []
        for document, chunk in candidates:
            score = vector_score(query, document, chunk)
            if score <= 0:
                continue
            hits.append(
                BackendSearchHit(
                    document=document,
                    chunk=chunk,
                    score=score,
                    backend=self.backend_name,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Describe the vector index sync operation for the given document."""

        return {
            "backend": self.backend_name,
            "table": self.table_name,
            "doc_id": document.id,
            "chunks_indexed": len(chunks),
        }
