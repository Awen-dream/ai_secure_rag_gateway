from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from app.domain.audit.models import AuditLog
from app.domain.chat.models import ChatMessage, ChatSession
from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.prompts.models import PromptTemplate
from app.domain.risk.models import PolicyDefinition
from app.domain.sources.models import SourceSyncRun


class SQLiteRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT,
                    source_connector TEXT,
                    source_document_id TEXT,
                    source_document_version TEXT,
                    owner_id TEXT NOT NULL,
                    department_scope TEXT NOT NULL,
                    visibility_scope TEXT NOT NULL,
                    security_level INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_error TEXT,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    current INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_no INTEGER NOT NULL,
                    section_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    security_level INTEGER NOT NULL,
                    department_scope TEXT NOT NULL,
                    role_scope TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    active_topic TEXT NOT NULL DEFAULT '',
                    permission_signature TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    token_usage INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id TEXT PRIMARY KEY,
                    scene TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    output_schema TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_by TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS policies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    high_risk_terms TEXT NOT NULL,
                    restricted_departments TEXT NOT NULL,
                    enabled INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    rewritten_query TEXT NOT NULL DEFAULT '',
                    scene TEXT NOT NULL DEFAULT '',
                    retrieval_docs_json TEXT NOT NULL,
                    prompt_json TEXT NOT NULL DEFAULT '{}',
                    risk_json TEXT NOT NULL DEFAULT '{}',
                    conversation_json TEXT NOT NULL DEFAULT '{}',
                    response_summary TEXT NOT NULL,
                    action TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_sync_runs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    triggered_by TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    continue_on_error INTEGER NOT NULL,
                    request_json TEXT NOT NULL,
                    result_items_json TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    imported_new INTEGER NOT NULL,
                    reused_current INTEGER NOT NULL,
                    created_new_version INTEGER NOT NULL,
                    queued INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute("DROP INDEX IF EXISTS idx_documents_tenant_content_hash")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_tenant_content_hash ON documents(tenant_id, content_hash)"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_source_ref
                ON documents(tenant_id, source_connector, source_document_id, current)
                """
            )
            existing_document_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "last_error" not in existing_document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN last_error TEXT")
            if "source_connector" not in existing_document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_connector TEXT")
            if "source_document_id" not in existing_document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_document_id TEXT")
            if "source_document_version" not in existing_document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_document_version TEXT")
            existing_session_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "active_topic" not in existing_session_columns:
                connection.execute("ALTER TABLE chat_sessions ADD COLUMN active_topic TEXT NOT NULL DEFAULT ''")
            if "permission_signature" not in existing_session_columns:
                connection.execute("ALTER TABLE chat_sessions ADD COLUMN permission_signature TEXT NOT NULL DEFAULT ''")
            existing_audit_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audit_logs)").fetchall()
            }
            if "rewritten_query" not in existing_audit_columns:
                connection.execute("ALTER TABLE audit_logs ADD COLUMN rewritten_query TEXT NOT NULL DEFAULT ''")
            if "scene" not in existing_audit_columns:
                connection.execute("ALTER TABLE audit_logs ADD COLUMN scene TEXT NOT NULL DEFAULT ''")
            if "prompt_json" not in existing_audit_columns:
                connection.execute("ALTER TABLE audit_logs ADD COLUMN prompt_json TEXT NOT NULL DEFAULT '{}'")
            if "risk_json" not in existing_audit_columns:
                connection.execute("ALTER TABLE audit_logs ADD COLUMN risk_json TEXT NOT NULL DEFAULT '{}'")
            if "conversation_json" not in existing_audit_columns:
                connection.execute("ALTER TABLE audit_logs ADD COLUMN conversation_json TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _dump_json(value) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _load_json(value: str):
        return json.loads(value) if value else []

    @staticmethod
    def _to_datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def find_document_by_content_hash(self, tenant_id: str, content_hash: str) -> Optional[DocumentRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND content_hash = ? ORDER BY current DESC, version DESC",
                (tenant_id, content_hash),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def find_current_document_by_source_ref(
        self,
        tenant_id: str,
        source_connector: str,
        source_document_id: str,
    ) -> Optional[DocumentRecord]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE tenant_id = ? AND source_connector = ? AND source_document_id = ? AND current = 1
                ORDER BY version DESC
                """,
                (tenant_id, source_connector, source_document_id),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents_by_title(self, tenant_id: str, title: str) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND title = ? ORDER BY version DESC",
                (tenant_id, title),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def list_documents_by_source_ref(
        self,
        tenant_id: str,
        source_connector: str,
        source_document_id: str,
    ) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE tenant_id = ? AND source_connector = ? AND source_document_id = ?
                ORDER BY version DESC
                """,
                (tenant_id, source_connector, source_document_id),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def save_document(self, document: DocumentRecord, chunks: list[DocumentChunk], previous_ids: list[str]) -> None:
        with self._connect() as connection:
            if previous_ids:
                connection.executemany("UPDATE documents SET current = 0 WHERE id = ?", [(doc_id,) for doc_id in previous_ids])
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, tenant_id, title, source_type, source_uri, source_connector, source_document_id, source_document_version,
                    owner_id, department_scope, visibility_scope, security_level, version, status, last_error, content_hash,
                    created_at, updated_at, tags, current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.tenant_id,
                    document.title,
                    document.source_type,
                    document.source_uri,
                    document.source_connector,
                    document.source_document_id,
                    document.source_document_version,
                    document.owner_id,
                    self._dump_json(document.department_scope),
                    self._dump_json(document.visibility_scope),
                    document.security_level,
                    document.version,
                    document.status.value,
                    document.last_error,
                    document.content_hash,
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                    self._dump_json(document.tags),
                    int(document.current),
                ),
            )
            connection.execute("DELETE FROM document_chunks WHERE doc_id = ?", (document.id,))
            connection.executemany(
                """
                INSERT INTO document_chunks (
                    id, doc_id, tenant_id, chunk_index, page_no, section_name, text, token_count,
                    security_level, department_scope, role_scope, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.tenant_id,
                        chunk.chunk_index,
                        chunk.page_no,
                        chunk.section_name,
                        chunk.text,
                        chunk.token_count,
                        chunk.security_level,
                        self._dump_json(chunk.department_scope),
                        self._dump_json(chunk.role_scope),
                        self._dump_json(chunk.metadata_json),
                    )
                    for chunk in chunks
                ],
            )

    def update_document(self, document: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, last_error = ?, updated_at = ?, current = ?, security_level = ?, department_scope = ?, visibility_scope = ?, tags = ?
                WHERE id = ?
                """,
                (
                    document.status.value,
                    document.last_error,
                    document.updated_at.isoformat(),
                    int(document.current),
                    document.security_level,
                    self._dump_json(document.department_scope),
                    self._dump_json(document.visibility_scope),
                    self._dump_json(document.tags),
                    document.id,
                ),
            )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, tenant_id: Optional[str] = None) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        params = ()
        if tenant_id:
            query += " WHERE tenant_id = ?"
            params = (tenant_id,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def list_chunks_for_document(self, doc_id: str) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_chunks_for_tenant(self, tenant_id: str) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_chunks WHERE tenant_id = ? ORDER BY doc_id, chunk_index ASC",
                (tenant_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def save_session(self, session: ChatSession) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO chat_sessions (
                    id, tenant_id, user_id, scene, status, summary, active_topic, permission_signature, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.tenant_id,
                    session.user_id,
                    session.scene,
                    session.status.value,
                    session.summary,
                    session.active_topic,
                    session.permission_signature,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self, tenant_id: str, user_id: str) -> list[ChatSession]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_sessions WHERE tenant_id = ? AND user_id = ? ORDER BY updated_at DESC",
                (tenant_id, user_id),
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def append_message(self, message: ChatMessage) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content, citations_json, token_usage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    self._dump_json([citation.model_dump() for citation in message.citations_json]),
                    message.token_usage,
                    message.created_at.isoformat(),
                ),
            )

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def save_prompt_template(self, template: PromptTemplate) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO prompt_templates (id, scene, version, name, content, output_schema, enabled, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    template.id,
                    template.scene,
                    template.version,
                    template.name,
                    template.content,
                    self._dump_json(template.output_schema),
                    int(template.enabled),
                    template.created_by,
                ),
            )

    def list_prompt_templates(self, scene: Optional[str] = None) -> list[PromptTemplate]:
        query = "SELECT * FROM prompt_templates"
        params = ()
        if scene:
            query += " WHERE scene = ?"
            params = (scene,)
        query += " ORDER BY scene, version DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_prompt_template(row) for row in rows]

    def save_policy(self, policy: PolicyDefinition) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO policies (id, name, description, high_risk_terms, restricted_departments, enabled)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.id,
                    policy.name,
                    policy.description,
                    self._dump_json(policy.high_risk_terms),
                    self._dump_json(policy.restricted_departments),
                    int(policy.enabled),
                ),
            )

    def list_policies(self) -> list[PolicyDefinition]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM policies ORDER BY name ASC").fetchall()
        return [self._row_to_policy(row) for row in rows]

    def append_audit_log(self, log: AuditLog) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (
                    id, user_id, tenant_id, session_id, request_id, query, rewritten_query, scene,
                    retrieval_docs_json, prompt_json, risk_json, conversation_json,
                    response_summary, action, risk_level, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.id,
                    log.user_id,
                    log.tenant_id,
                    log.session_id,
                    log.request_id,
                    log.query,
                    log.rewritten_query,
                    log.scene,
                    self._dump_json(log.retrieval_docs_json),
                    self._dump_json(log.prompt_json),
                    self._dump_json(log.risk_json),
                    self._dump_json(log.conversation_json),
                    log.response_summary,
                    log.action,
                    log.risk_level,
                    log.latency_ms,
                    log.created_at.isoformat(),
                ),
            )

    def list_audit_logs(self) -> list[AuditLog]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_logs ORDER BY created_at DESC").fetchall()
        return [self._row_to_audit_log(row) for row in rows]

    def append_source_sync_run(self, run: SourceSyncRun) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_sync_runs (
                    id, tenant_id, provider, triggered_by, mode, continue_on_error, request_json, result_items_json,
                    total, succeeded, failed, imported_new, reused_current, created_new_version, queued, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.tenant_id,
                    run.provider,
                    run.triggered_by,
                    run.mode,
                    int(run.continue_on_error),
                    self._dump_json(run.request_json),
                    self._dump_json(run.result_items_json),
                    run.total,
                    run.succeeded,
                    run.failed,
                    run.imported_new,
                    run.reused_current,
                    run.created_new_version,
                    run.queued,
                    run.status,
                    run.created_at.isoformat(),
                ),
            )

    def list_source_sync_runs(self, tenant_id: str, provider: str) -> list[SourceSyncRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_sync_runs
                WHERE tenant_id = ? AND provider = ?
                ORDER BY created_at DESC
                """,
                (tenant_id, provider),
            ).fetchall()
        return [self._row_to_source_sync_run(row) for row in rows]

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            title=row["title"],
            source_type=row["source_type"],
            source_uri=row["source_uri"],
            source_connector=row["source_connector"],
            source_document_id=row["source_document_id"],
            source_document_version=row["source_document_version"],
            owner_id=row["owner_id"],
            department_scope=self._load_json(row["department_scope"]),
            visibility_scope=self._load_json(row["visibility_scope"]),
            security_level=row["security_level"],
            version=row["version"],
            status=row["status"],
            last_error=row["last_error"],
            content_hash=row["content_hash"],
            created_at=self._to_datetime(row["created_at"]),
            updated_at=self._to_datetime(row["updated_at"]),
            tags=self._load_json(row["tags"]),
            current=bool(row["current"]),
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> DocumentChunk:
        return DocumentChunk(
            id=row["id"],
            doc_id=row["doc_id"],
            tenant_id=row["tenant_id"],
            chunk_index=row["chunk_index"],
            page_no=row["page_no"],
            section_name=row["section_name"],
            text=row["text"],
            token_count=row["token_count"],
            security_level=row["security_level"],
            department_scope=self._load_json(row["department_scope"]),
            role_scope=self._load_json(row["role_scope"]),
            metadata_json=self._load_json(row["metadata_json"]),
        )

    def _row_to_session(self, row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            scene=row["scene"],
            status=row["status"],
            summary=row["summary"],
            active_topic=row["active_topic"],
            permission_signature=row["permission_signature"],
            created_at=self._to_datetime(row["created_at"]),
            updated_at=self._to_datetime(row["updated_at"]),
        )

    def _row_to_message(self, row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            citations_json=self._load_json(row["citations_json"]),
            token_usage=row["token_usage"],
            created_at=self._to_datetime(row["created_at"]),
        )

    def _row_to_prompt_template(self, row: sqlite3.Row) -> PromptTemplate:
        return PromptTemplate(
            id=row["id"],
            scene=row["scene"],
            version=row["version"],
            name=row["name"],
            content=row["content"],
            output_schema=self._load_json(row["output_schema"]),
            enabled=bool(row["enabled"]),
            created_by=row["created_by"],
        )

    def _row_to_policy(self, row: sqlite3.Row) -> PolicyDefinition:
        return PolicyDefinition(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            high_risk_terms=self._load_json(row["high_risk_terms"]),
            restricted_departments=self._load_json(row["restricted_departments"]),
            enabled=bool(row["enabled"]),
        )

    def _row_to_audit_log(self, row: sqlite3.Row) -> AuditLog:
        return AuditLog(
            id=row["id"],
            user_id=row["user_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            request_id=row["request_id"],
            query=row["query"],
            rewritten_query=row["rewritten_query"],
            scene=row["scene"],
            retrieval_docs_json=self._load_json(row["retrieval_docs_json"]),
            prompt_json=self._load_json(row["prompt_json"]),
            risk_json=self._load_json(row["risk_json"]),
            conversation_json=self._load_json(row["conversation_json"]),
            response_summary=row["response_summary"],
            action=row["action"],
            risk_level=row["risk_level"],
            latency_ms=row["latency_ms"],
            created_at=self._to_datetime(row["created_at"]),
        )

    def _row_to_source_sync_run(self, row: sqlite3.Row) -> SourceSyncRun:
        return SourceSyncRun(
            id=row["id"],
            tenant_id=row["tenant_id"],
            provider=row["provider"],
            triggered_by=row["triggered_by"],
            mode=row["mode"],
            continue_on_error=bool(row["continue_on_error"]),
            request_json=self._load_json(row["request_json"]),
            result_items_json=self._load_json(row["result_items_json"]),
            total=row["total"],
            succeeded=row["succeeded"],
            failed=row["failed"],
            imported_new=row["imported_new"],
            reused_current=row["reused_current"],
            created_new_version=row["created_new_version"],
            queued=row["queued"],
            status=row["status"],
            created_at=self._to_datetime(row["created_at"]),
        )
