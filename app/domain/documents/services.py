from __future__ import annotations

import hashlib
import uuid
from datetime import datetime

from app.application.ingestion.pipelines import chunk_text
from app.domain.auth.models import UserContext
from app.domain.auth.policies import can_access_department
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.documents.schemas import DocumentUploadRequest
from app.infrastructure.db.repositories.memory import store


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentService:
    def upload_document(self, payload: DocumentUploadRequest, user: UserContext) -> DocumentRecord:
        content_hash = hashlib.sha256(payload.content.strip().encode("utf-8")).hexdigest()
        if content_hash in store.content_hashes:
            return store.documents[store.content_hashes[content_hash]]

        previous_versions = [
            doc
            for doc in store.documents.values()
            if doc.tenant_id == user.tenant_id and doc.title == payload.title
        ]
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

        chunks = [
            DocumentChunk(
                id=f"chunk_{uuid.uuid4().hex[:12]}",
                doc_id=document.id,
                tenant_id=document.tenant_id,
                chunk_index=index,
                section_name=f"Section {index + 1}",
                text=chunk,
                token_count=max(len(chunk.split()), 1),
                security_level=document.security_level,
                department_scope=document.department_scope,
                metadata_json={"title": document.title},
            )
            for index, chunk in enumerate(chunk_text(payload.content))
        ]

        document.status = DocumentStatus.SUCCESS
        store.documents[document.id] = document
        store.document_chunks[document.id] = chunks
        store.content_hashes[content_hash] = document.id
        return document

    def list_documents(self, user: UserContext) -> list[DocumentRecord]:
        return [
            document
            for document in store.documents.values()
            if document.tenant_id == user.tenant_id and self._has_document_access(document, user)
        ]

    def reindex_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        document = self.get_document(doc_id, user)
        document.status = DocumentStatus.INDEXING
        document.updated_at = utcnow()
        document.status = DocumentStatus.SUCCESS
        return document

    def get_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        document = store.documents.get(doc_id)
        if not document or document.tenant_id != user.tenant_id:
            raise KeyError(doc_id)
        if not self._has_document_access(document, user):
            raise PermissionError(doc_id)
        return document

    def get_accessible_chunks(self, user: UserContext) -> list[tuple[DocumentRecord, DocumentChunk]]:
        results: list[tuple[DocumentRecord, DocumentChunk]] = []
        for doc_id, document in store.documents.items():
            if document.tenant_id != user.tenant_id or not self._has_document_access(document, user):
                continue
            for chunk in store.document_chunks.get(doc_id, []):
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
