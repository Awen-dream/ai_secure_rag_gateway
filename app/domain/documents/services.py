from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from app.application.ingestion.pipelines import chunk_document
from app.domain.auth.models import UserContext
from app.domain.auth.policies import can_access_department
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.sqlite import SQLiteRepository


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentService:
    """Owns document lifecycle, permission-aware reads and retrieval index synchronization."""

    def __init__(
        self,
        repository: SQLiteRepository,
        indexing_service: Optional[RetrievalIndexingService] = None,
    ) -> None:
        self.repository = repository
        self.indexing_service = indexing_service

    def upload_document(self, payload: DocumentUploadRequest, user: UserContext) -> DocumentRecord:
        """Create a new document version, persist chunks and refresh retrieval indexes."""

        content_hash = hashlib.sha256(payload.content.strip().encode("utf-8")).hexdigest()
        existing_document = self.repository.find_document_by_content_hash(user.tenant_id, content_hash)
        if existing_document:
            return existing_document

        previous_versions = self.repository.list_documents_by_title(user.tenant_id, payload.title)
        version = max((doc.version for doc in previous_versions), default=0) + 1
        now = utcnow()

        document = DocumentRecord(
            id=f"doc_{uuid.uuid4().hex[:12]}",
            tenant_id=user.tenant_id,
            title=payload.title,
            source_type=payload.source_type,
            source_uri=payload.source_uri,
            owner_id=payload.owner_id or user.user_id,
            department_scope=payload.department_scope or [user.department_id],
            visibility_scope=payload.visibility_scope,
            security_level=payload.security_level,
            version=version,
            status=DocumentStatus.INDEXING,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            tags=payload.tags,
            current=True,
        )

        for record in previous_versions:
            record.current = False

        chunk_payloads = chunk_document(payload.content)
        chunks = [
            DocumentChunk(
                id=f"chunk_{uuid.uuid4().hex[:12]}",
                doc_id=document.id,
                tenant_id=document.tenant_id,
                chunk_index=index,
                section_name=chunk.section_name,
                text=chunk.text,
                token_count=chunk.token_count,
                security_level=document.security_level,
                department_scope=document.department_scope,
                metadata_json={
                    "title": document.title,
                    "section_name": chunk.section_name,
                    "heading_path": " > ".join(chunk.heading_path),
                },
            )
            for index, chunk in enumerate(chunk_payloads)
        ]

        document.status = DocumentStatus.SUCCESS
        self.repository.save_document(document, chunks, [record.id for record in previous_versions])
        if self.indexing_service:
            self.indexing_service.upsert_document(document, chunks)
        return document

    def list_documents(self, user: UserContext) -> list[DocumentRecord]:
        """List current documents visible under the caller's tenant and permission scope."""

        return [
            document
            for document in self.repository.list_documents(user.tenant_id)
            if document.tenant_id == user.tenant_id and self._has_document_access(document, user)
        ]

    def reindex_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        """Refresh a document's retrieval indexes after metadata or content-level changes."""

        document = self.get_document(doc_id, user)
        document.status = DocumentStatus.INDEXING
        document.updated_at = utcnow()
        document.status = DocumentStatus.SUCCESS
        self.repository.update_document(document)
        if self.indexing_service:
            self.indexing_service.upsert_document(document, self.repository.list_chunks_for_document(doc_id))
        return document

    def get_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        """Load one document and enforce tenant plus permission boundaries."""

        document = self.repository.get_document(doc_id)
        if not document or document.tenant_id != user.tenant_id:
            raise KeyError(doc_id)
        if not self._has_document_access(document, user):
            raise PermissionError(doc_id)
        return document

    def get_accessible_chunks(self, user: UserContext) -> list[tuple[DocumentRecord, DocumentChunk]]:
        """Return only those chunks that are allowed to participate in retrieval."""

        results: list[tuple[DocumentRecord, DocumentChunk]] = []
        documents = {document.id: document for document in self.repository.list_documents(user.tenant_id)}
        for chunk in self.repository.list_chunks_for_tenant(user.tenant_id):
            document = documents.get(chunk.doc_id)
            if not document:
                continue
            if document.tenant_id != user.tenant_id or not self._has_document_access(document, user):
                continue
            if self._has_chunk_access(chunk, user):
                results.append((document, chunk))
        return results

    @staticmethod
    def _has_document_access(document: DocumentRecord, user: UserContext) -> bool:
        return (
            document.current
            and user.clearance_level >= document.security_level
            and can_access_department(document.department_scope, user)
        )

    @staticmethod
    def _has_chunk_access(chunk: DocumentChunk, user: UserContext) -> bool:
        role_ok = not chunk.role_scope or user.role in chunk.role_scope
        return role_ok and user.clearance_level >= chunk.security_level and can_access_department(
            chunk.department_scope, user
        )
