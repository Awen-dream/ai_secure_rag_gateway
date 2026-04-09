from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Sequence

from app.application.access.service import AccessFilter
from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalBackendHealth, RetrievalBackendInfo
from app.domain.retrieval.retrievers import semantic_features, vector_score
from app.infrastructure.llm.openai_embeddings import OpenAIEmbeddingClient

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
        embedding_client: OpenAIEmbeddingClient | None = None,
    ) -> None:
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.mode = mode
        self.dsn = dsn
        self.auto_init_schema = auto_init_schema
        self.embedding_client = embedding_client

    def search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None = None,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
    ) -> list[BackendSearchHit]:
        """Return semantic hits scored with a lightweight local similarity fallback."""

        if self.can_execute() and candidates:
            try:
                return self._execute_search(
                    query,
                    candidates,
                    top_k,
                    access_filter,
                    tag_filters=tag_filters,
                    year_filters=year_filters,
                )
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

        rows = self.build_upsert_rows(document, chunks, use_runtime_embeddings=self.can_execute())
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

    def delete_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Remove all stored embeddings for one document."""

        executed = False
        if self.can_execute():
            self._execute_delete(document.id)
            executed = True

        return {
            "backend": self.backend_name,
            "table": self.table_name,
            "doc_id": document.id,
            "chunks_deleted": len(chunks),
            "delete_sql_preview": self.build_delete_sql(),
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
                "embedding_provider": self.embedding_client.provider if self.embedding_client else "local-fallback",
                "embedding_execute_enabled": bool(self.embedding_client and self.embedding_client.can_execute()),
            },
            capabilities=[
                "vector_search",
                "cosine_distance",
                "metadata_filtering",
                "upsert_preview",
                "ddl_preview",
                "remote_embeddings",
            ],
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
    owner_id TEXT NOT NULL,
    section_name TEXT NOT NULL,
    content TEXT NOT NULL,
    department_scope JSONB NOT NULL,
    visibility_scope JSONB NOT NULL,
    role_scope JSONB NOT NULL,
    security_level INTEGER NOT NULL,
    current BOOLEAN NOT NULL,
    status TEXT NOT NULL,
    metadata_json JSONB NOT NULL,
    tags JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    embedding VECTOR({self.embedding_dimension}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant_id ON {self.table_name}(tenant_id);
CREATE INDEX IF NOT EXISTS idx_{self.table_name}_doc_id ON {self.table_name}(doc_id);
CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding_cosine
ON {self.table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
""".strip()

    def build_upsert_sql(self) -> str:
        """Return the SQL template used for chunk embedding upserts."""

        return f"""
INSERT INTO {self.table_name} (
    chunk_id, doc_id, tenant_id, title, owner_id, section_name, content,
    department_scope, visibility_scope, role_scope, security_level, current, status, metadata_json, tags, created_at, updated_at, embedding
) VALUES (
    %(chunk_id)s, %(doc_id)s, %(tenant_id)s, %(title)s, %(owner_id)s, %(section_name)s, %(content)s,
    %(department_scope)s::jsonb, %(visibility_scope)s::jsonb, %(role_scope)s::jsonb, %(security_level)s, %(current)s, %(status)s, %(metadata_json)s::jsonb, %(tags)s::jsonb, %(created_at)s, %(updated_at)s, %(embedding)s::vector
)
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    tenant_id = EXCLUDED.tenant_id,
    title = EXCLUDED.title,
    owner_id = EXCLUDED.owner_id,
    section_name = EXCLUDED.section_name,
    content = EXCLUDED.content,
    department_scope = EXCLUDED.department_scope,
    visibility_scope = EXCLUDED.visibility_scope,
    role_scope = EXCLUDED.role_scope,
    security_level = EXCLUDED.security_level,
    current = EXCLUDED.current,
    status = EXCLUDED.status,
    metadata_json = EXCLUDED.metadata_json,
    tags = EXCLUDED.tags,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at,
    embedding = EXCLUDED.embedding;
""".strip()

    def build_upsert_rows(
        self,
        document: DocumentRecord,
        chunks: Sequence[DocumentChunk],
        use_runtime_embeddings: bool = False,
    ) -> list[dict]:
        """Build parameter rows for pgvector upserts using remote or deterministic embeddings."""

        embeddings = self._embed_texts(
            [chunk.text for chunk in chunks],
            use_runtime_embeddings=use_runtime_embeddings,
        )
        return [
            {
                "chunk_id": chunk.id,
                "doc_id": document.id,
                "tenant_id": document.tenant_id,
                "title": document.title,
                "owner_id": document.owner_id,
                "section_name": chunk.section_name,
                "content": chunk.text,
                "department_scope": json.dumps(chunk.department_scope, ensure_ascii=False),
                "visibility_scope": json.dumps(document.visibility_scope, ensure_ascii=False),
                "role_scope": json.dumps(chunk.role_scope, ensure_ascii=False),
                "security_level": chunk.security_level,
                "current": document.current,
                "status": document.status.value,
                "metadata_json": json.dumps(chunk.metadata_json, ensure_ascii=False),
                "tags": json.dumps([tag.lower() for tag in document.tags], ensure_ascii=False),
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "embedding": self._to_pgvector_literal(embeddings[index]),
            }
            for index, chunk in enumerate(chunks)
        ]

    def build_search_sql(
        self,
        access_filter: AccessFilter,
        top_k: int,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
    ) -> str:
        """Return the SQL template for permission-aware pgvector search."""

        extra_where = ""
        if tag_filters:
            extra_where += " AND tags ?| %(tag_filters)s::text[]"
        if year_filters:
            extra_where += (
                " AND (EXTRACT(YEAR FROM updated_at) = ANY(%(year_filters)s::int[]) "
                "OR EXTRACT(YEAR FROM created_at) = ANY(%(year_filters)s::int[]))"
            )
        return f"""
SELECT
    chunk_id,
    doc_id,
    tenant_id,
    title,
    owner_id,
    section_name,
    content,
    department_scope,
    visibility_scope,
    role_scope,
    security_level,
    current,
    status,
    metadata_json,
    tags,
    created_at,
    updated_at,
    1 - (embedding <=> %(query_embedding)s::vector) AS similarity_score
FROM {self.table_name}
WHERE {access_filter.build_pgvector_where_clause()}{extra_where}
ORDER BY embedding <=> %(query_embedding)s
LIMIT {top_k};
""".strip()

    def build_delete_sql(self) -> str:
        """Return the SQL template used to delete one document from pgvector storage."""

        return f"DELETE FROM {self.table_name} WHERE doc_id = %(doc_id)s;"

    def _execute_upsert(self, rows: Sequence[dict]) -> None:
        """Execute pgvector upserts against a real PostgreSQL backend when configured."""

        if not self.can_execute() or not rows:
            return
        sql = self.build_upsert_sql()
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(sql, rows)

    def _execute_delete(self, doc_id: str) -> None:
        """Execute one pgvector document delete when the backend is enabled."""

        if not self.can_execute():
            return
        with psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(self.build_delete_sql(), {"doc_id": doc_id})

    def _execute_search(
        self,
        query: str,
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
    ) -> list[BackendSearchHit]:
        """Execute a real pgvector similarity query and map results back to allowed candidates."""

        if not self.can_execute():
            return []
        if access_filter is None:
            raise RuntimeError("Access filter is required for remote pgvector search.")

        candidate_lookup = {chunk.id: (document, chunk) for document, chunk in candidates}
        sql = self.build_search_sql(access_filter, top_k, tag_filters=tag_filters, year_filters=year_filters)
        params = access_filter.build_pgvector_params(list(candidate_lookup.keys()))
        params["tag_filters"] = [tag.lower() for tag in tag_filters or []]
        params["year_filters"] = list(year_filters or [])
        params["query_embedding"] = self._to_pgvector_literal(
            self._embed_text(query, use_runtime_embeddings=self.can_execute())
        )
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

    def _embed_text(self, text: str, use_runtime_embeddings: bool = False) -> list[float]:
        """Generate one embedding via the configured provider or deterministic local fallback."""

        return self._embed_texts([text], use_runtime_embeddings=use_runtime_embeddings)[0]

    def _embed_texts(self, texts: Sequence[str], use_runtime_embeddings: bool = False) -> list[list[float]]:
        """Generate embeddings for multiple texts via the configured provider or local fallback."""

        if use_runtime_embeddings and self.embedding_client and self.embedding_client.can_execute():
            try:
                vectors = self.embedding_client.embed_texts(texts)
                if vectors:
                    return vectors
            except Exception:
                pass
        return [self._local_embed_text(text) for text in texts]

    def _local_embed_text(self, text: str) -> list[float]:
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
