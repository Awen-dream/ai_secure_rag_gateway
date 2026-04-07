from __future__ import annotations

import json
from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalBackendInfo
from app.domain.retrieval.retrievers import vector_score


class PGVectorStore:
    """Development-safe PGVector adapter.

    The adapter exposes the same methods we expect from a real PostgreSQL + pgvector
    implementation. For now it computes semantic similarity locally, which keeps the
    hybrid retrieval flow runnable in tests and local development.
    """

    backend_name = "pgvector"

    def __init__(self, table_name: str = "document_embeddings", embedding_dimension: int = 1536, mode: str = "local-fallback") -> None:
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.mode = mode

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
            "upsert_sql_preview": self.build_upsert_sql(),
        }

    def describe_backend(self) -> RetrievalBackendInfo:
        """Return deployment and capability metadata for the vector backend."""

        return RetrievalBackendInfo(
            backend=self.backend_name,
            mode=self.mode,
            config={
                "table_name": self.table_name,
                "embedding_dimension": self.embedding_dimension,
            },
            capabilities=["vector_search", "cosine_distance", "metadata_filtering", "upsert_preview"],
        )

    def build_table_ddl(self) -> str:
        """Return the SQL DDL required for a production pgvector table."""

        return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {self.table_name} (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    title TEXT NOT NULL,
    section_name TEXT NOT NULL,
    content TEXT NOT NULL,
    department_scope JSONB NOT NULL,
    role_scope JSONB NOT NULL,
    security_level INTEGER NOT NULL,
    metadata_json JSONB NOT NULL,
    embedding VECTOR({self.embedding_dimension}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant_id ON {self.table_name}(tenant_id);
CREATE INDEX IF NOT EXISTS idx_{self.table_name}_doc_id ON {self.table_name}(doc_id);
""".strip()

    def build_upsert_sql(self) -> str:
        """Return the SQL template used for chunk embedding upserts."""

        return f"""
INSERT INTO {self.table_name} (
    chunk_id, doc_id, tenant_id, title, section_name, content,
    department_scope, role_scope, security_level, metadata_json, embedding
) VALUES (
    %(chunk_id)s, %(doc_id)s, %(tenant_id)s, %(title)s, %(section_name)s, %(content)s,
    %(department_scope)s::jsonb, %(role_scope)s::jsonb, %(security_level)s, %(metadata_json)s::jsonb, %(embedding)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    tenant_id = EXCLUDED.tenant_id,
    title = EXCLUDED.title,
    section_name = EXCLUDED.section_name,
    content = EXCLUDED.content,
    department_scope = EXCLUDED.department_scope,
    role_scope = EXCLUDED.role_scope,
    security_level = EXCLUDED.security_level,
    metadata_json = EXCLUDED.metadata_json,
    embedding = EXCLUDED.embedding;
""".strip()

    def build_upsert_rows(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> list[dict]:
        """Build parameter rows for a future real pgvector upsert implementation."""

        return [
            {
                "chunk_id": chunk.id,
                "doc_id": document.id,
                "tenant_id": document.tenant_id,
                "title": document.title,
                "section_name": chunk.section_name,
                "content": chunk.text,
                "department_scope": json.dumps(chunk.department_scope, ensure_ascii=False),
                "role_scope": json.dumps(chunk.role_scope, ensure_ascii=False),
                "security_level": chunk.security_level,
                "metadata_json": json.dumps(chunk.metadata_json, ensure_ascii=False),
                "embedding": f"<{self.embedding_dimension}-dim-vector>",
            }
            for chunk in chunks
        ]

    def build_search_sql(self, tenant_id: str, top_k: int) -> str:
        """Return the SQL template for permission-aware pgvector search."""

        return f"""
SELECT
    chunk_id,
    doc_id,
    tenant_id,
    title,
    section_name,
    content,
    department_scope,
    role_scope,
    security_level,
    metadata_json,
    1 - (embedding <=> %(query_embedding)s) AS similarity_score
FROM {self.table_name}
WHERE tenant_id = %(tenant_id)s
ORDER BY embedding <=> %(query_embedding)s
LIMIT {top_k};
""".strip()
