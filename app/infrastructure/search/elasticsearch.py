from __future__ import annotations

import json
from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalBackendInfo
from app.domain.retrieval.retrievers import keyword_features


class ElasticsearchSearch:
    """Development-safe Elasticsearch adapter.

    The public interface is intentionally aligned with the real backend contract we want
    to keep later. In local development it scores permission-filtered candidates
    in-process, which lets the hybrid retrieval pipeline stay runnable without an
    external Elasticsearch cluster.
    """

    backend_name = "elasticsearch"

    def __init__(self, index_name: str = "knowledge_chunks", mode: str = "local-fallback") -> None:
        self.index_name = index_name
        self.mode = mode

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
            "bulk_preview_lines": len(self.build_bulk_payload(document, chunks).splitlines()),
        }

    def describe_backend(self) -> RetrievalBackendInfo:
        """Return deployment and capability metadata for the keyword backend."""

        return RetrievalBackendInfo(
            backend=self.backend_name,
            mode=self.mode,
            config={"index_name": self.index_name},
            capabilities=["keyword_search", "bm25_style_matching", "metadata_filtering", "bulk_index_preview"],
        )

    def build_index_mapping(self) -> dict:
        """Return the target Elasticsearch mapping we expect for chunk indexing."""

        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {"type": "standard"},
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "section_name": {"type": "text"},
                    "content": {"type": "text"},
                    "department_scope": {"type": "keyword"},
                    "role_scope": {"type": "keyword"},
                    "security_level": {"type": "integer"},
                    "metadata_json": {"type": "object", "enabled": True},
                }
            },
        }

    def build_bulk_payload(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> str:
        """Build a newline-delimited JSON bulk payload preview for document chunk indexing."""

        lines: list[str] = []
        for chunk in chunks:
            lines.append(json.dumps({"index": {"_index": self.index_name, "_id": chunk.id}}, ensure_ascii=False))
            lines.append(
                json.dumps(
                    {
                        "chunk_id": chunk.id,
                        "doc_id": document.id,
                        "tenant_id": document.tenant_id,
                        "title": document.title,
                        "section_name": chunk.section_name,
                        "content": chunk.text,
                        "department_scope": chunk.department_scope,
                        "role_scope": chunk.role_scope,
                        "security_level": chunk.security_level,
                        "metadata_json": chunk.metadata_json,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)

    def build_search_body(self, query: str, tenant_id: str, terms: Sequence[str], top_k: int) -> dict:
        """Build the Elasticsearch DSL body for a permission-aware keyword search."""

        should_clauses = [
            {"match": {"title": {"query": query, "boost": 3}}},
            {"match": {"content": {"query": query, "boost": 1}}},
        ]
        for term in terms[:8]:
            should_clauses.append({"term": {"title": {"value": term, "boost": 2}}})

        return {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": [{"term": {"tenant_id": tenant_id}}],
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }
