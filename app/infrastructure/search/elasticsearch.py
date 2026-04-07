from __future__ import annotations

from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.retrievers import keyword_features


class ElasticsearchSearch:
    """Development-safe Elasticsearch adapter.

    The public interface is intentionally aligned with the real backend contract we want
    to keep later. In local development it scores permission-filtered candidates
    in-process, which lets the hybrid retrieval pipeline stay runnable without an
    external Elasticsearch cluster.
    """

    backend_name = "elasticsearch"

    def __init__(self, index_name: str = "knowledge_chunks") -> None:
        self.index_name = index_name

    def search(
        self,
        query: str,
        terms: Sequence[str],
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
    ) -> list[BackendSearchHit]:
        """Return keyword-oriented hits using title and body term matching."""

        hits: list[BackendSearchHit] = []
        for document, chunk in candidates:
            score, matched_terms = keyword_features(list(terms), document, chunk)
            if score <= 0:
                continue
            hits.append(
                BackendSearchHit(
                    document=document,
                    chunk=chunk,
                    score=score,
                    backend=self.backend_name,
                    matched_terms=matched_terms,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Describe the keyword index sync operation for the given document."""

        return {
            "backend": self.backend_name,
            "index": self.index_name,
            "doc_id": document.id,
            "chunks_indexed": len(chunks),
        }
