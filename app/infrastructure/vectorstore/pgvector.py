from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Sequence

from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalBackendHealth, RetrievalBackendInfo
from app.domain.retrieval.retrievers import semantic_features, vector_score

try:
    import psycopg
except ImportError:  # pragma: no cover - optional runtime dependency
    psycopg = None


class PGVectorStore:
    """Development-safe PGVector adapter.

    The adapter exposes the same methods we expect from a real PostgreSQL + pgvector
    implementation. For now it computes semantic similarity locally, which keeps the
    hybrid retrieval flow runnable in tests and local development.
    """

    backend_name = "pgvector"

    def __init__(
        self,
        table_name: str = "document_embeddings",
        embedding_dimension: int = 1536,
        mode: str = "local-fallback",
        dsn: str | None = None,
        auto_init_schema: bool = False,
    ) -> None:
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.dsn = dsn
        self.auto_init_schema = auto_init_schema

    def search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
    ) -> list[BackendSearchHit]:
        """Return semantic hits scored with a lightweight local similarity fallback."""

        if self.can_execute() and candidates:
            try:
                return self._execute_search(query, candidates, top_k)
            except Exception:
                pass

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
        """Sync or preview vector index updates for the given document."""

        rows = self.build_upsert_rows(document, chunks)
        executed = False
        if self.can_execute() and rows:
            if self.auto_init_schema:
                self.initialize_schema()
            self._execute_upsert(rows)
            executed = True

        return {
            "backend": self.backend_name,
            "table": self.table_name,
            "doc_id": document.id,
            "chunks_indexed": len(chunks),
            "upsert_sql_preview": self.build_upsert_sql(),
            "executed": executed,
        }

    def describe_backend(self) -> RetrievalBackendInfo:
        """Return deployment and capability metadata for the vector backend."""

        return RetrievalBackendInfo(
            backend=self.backend_name,
            mode=self.mode,
            config={
                "table_name": self.table_name,
                "embedding_dimension": self.embedding_dimension,
                "dsn": self.dsn,
                "auto_init_schema": self.auto_init_schema,
            },
            capabilities=["vector_search", "cosine_distance", "metadata_filtering", "upsert_preview", "ddl_preview"],
        )

    def can_execute(self) -> bool:
        """Return whether this adapter can talk to a real PostgreSQL + pgvector backend."""

        return self.mode == "postgres" and bool(self.dsn) and psycopg is not None

    def health_check(self) -> RetrievalBackendHealth:
        """Return reachability status for the configured PostgreSQL + pgvector backend."""

        reachable = False
        detail = {"dsn": self.dsn, "table_name": self.table_name}
        if self.can_execute():
            try:
                with psycopg.connect(self.dsn) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                reachable = True
            except Exception as exc:
                detail["error"] = str(exc)
        return RetrievalBackendHealth(
            backend=self.backend_name,
            execute_enabled=self.can_execute(),
            reachable=reachable,
            detail=detail,
        )

    def initialize_schema(self) -> dict:
        """Create pgvector extension and table when real execution mode is enabled."""

        ddl = self.build_table_ddl()
        executed = False
        if self.can_execute():
            with psycopg.connect(self.dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(ddl)
            executed = True
        return {"backend": self.backend_name, "executed": executed, "ddl": ddl}

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
        """Build parameter rows for pgvector upserts using deterministic local embeddings."""

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
                "embedding": self._to_pgvector_literal(self._embed_text(chunk.text)),
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
  AND chunk_id = ANY(%(chunk_ids)s::text[])
ORDER BY embedding <=> %(query_embedding)s
LIMIT {top_k};
""".strip()

    def _execute_upsert(self, rows: Sequence[dict]) -> None:
        """Execute pgvector upserts against a real PostgreSQL backend when configured."""

        if not self.can_execute() or not rows:
            return
        sql = self.build_upsert_sql()
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(sql, rows)

    def _execute_search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
    ) -> list[BackendSearchHit]:
        """Execute a real pgvector similarity query and map results back to allowed candidates."""

        if not self.can_execute():
            return []

        candidate_lookup = {chunk.id: (document, chunk) for document, chunk in candidates}
        tenant_id = candidates[0][0].tenant_id
        sql = self.build_search_sql(tenant_id, top_k)
        params = {
            "tenant_id": tenant_id,
            "query_embedding": self._to_pgvector_literal(self._embed_text(query)),
            "chunk_ids": list(candidate_lookup.keys()),
        }
        hits: list[BackendSearchHit] = []
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        for row in rows:
            chunk_id = row[0]
            if chunk_id not in candidate_lookup:
                continue
            document, chunk = candidate_lookup[chunk_id]
            hits.append(
                BackendSearchHit(
                    document=document,
                    chunk=chunk,
                    score=float(row[-1]),
                    backend=self.backend_name,
                )
            )
        return hits

    def _embed_text(self, text: str) -> list[float]:
        """Generate a deterministic local embedding so pgvector integration can be tested offline."""

        vector = [0.0] * self.embedding_dimension
        features = semantic_features(text)
        for token, weight in features.items():
            digest = sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.embedding_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign * float(weight)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]

    @staticmethod
    def _to_pgvector_literal(values: Sequence[float]) -> str:
        """Render a Python float vector into pgvector text literal syntax."""

        return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
