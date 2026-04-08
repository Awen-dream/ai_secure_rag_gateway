from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.auth.filter_builder import build_access_filter
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentService:
    """Owns document lifecycle, permission-aware reads and retrieval index synchronization."""

    def __init__(
        self,
        repository: MetadataRepository,
        indexing_service: Optional[RetrievalIndexingService] = None,
        source_store: Optional[LocalDocumentSourceStore] = None,
        ingestion_orchestrator: Optional[DocumentIngestionOrchestrator] = None,
    ) -> None:
        self.repository = repository
        self.indexing_service = indexing_service
        self.source_store = source_store
        self.ingestion_orchestrator = ingestion_orchestrator

    def upload_document(self, payload: DocumentUploadRequest, user: UserContext) -> DocumentRecord:
        """Register one text upload and persist the staged source for later ingestion."""

        return self._register_document(
            payload=payload,
            user=user,
            source_bytes=payload.content.encode("utf-8"),
            process_async=payload.async_mode,
        )

    def upload_document_file(
        self,
        payload: DocumentUploadRequest,
        user: UserContext,
        file_bytes: bytes,
        process_async: bool = False,
    ) -> DocumentRecord:
        """Register one file upload and persist the staged source for later ingestion."""

        return self._register_document(
            payload=payload,
            user=user,
            source_bytes=file_bytes,
            process_async=process_async,
        )

    def _register_document(
        self,
        payload: DocumentUploadRequest,
        user: UserContext,
        source_bytes: bytes,
        process_async: bool,
    ) -> DocumentRecord:
        """Create a pending document version and stage its source bytes for the ingestion worker."""

        normalized_source_bytes = source_bytes if source_bytes else payload.content.strip().encode("utf-8")
        content_hash = hashlib.sha256(normalized_source_bytes).hexdigest()
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
            status=DocumentStatus.PENDING,
            last_error=None,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            tags=payload.tags,
            current=False,
        )

        if self.source_store:
            self.source_store.save_source(document.id, document.source_type, normalized_source_bytes)

        self.repository.save_document(document, [], [])
        return document

    def list_documents(self, user: UserContext) -> list[DocumentRecord]:
        """List current documents visible under the caller's tenant and permission scope."""

        access_filter = build_access_filter(user)
        return [
            document
            for document in self.repository.list_documents(user.tenant_id)
            if access_filter.allows_document(document)
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

    def retry_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        """Reset one failed document to pending so the background ingestion flow can try again."""

        document = self.get_document(doc_id, user)
        if not self.ingestion_orchestrator:
            raise RuntimeError("Document ingestion orchestrator is not configured.")
        return self.ingestion_orchestrator.retry_document(document.id)

    def get_document(self, doc_id: str, user: UserContext) -> DocumentRecord:
        """Load one document and enforce tenant plus permission boundaries."""

        document = self.repository.get_document(doc_id)
        if not document or document.tenant_id != user.tenant_id:
            raise KeyError(doc_id)
        if not build_access_filter(user).allows_document(document):
            raise PermissionError(doc_id)
        return document

    def get_accessible_chunks(self, user: UserContext) -> list[tuple[DocumentRecord, DocumentChunk]]:
        """Return only those chunks that are allowed to participate in retrieval."""

        results: list[tuple[DocumentRecord, DocumentChunk]] = []
        access_filter = build_access_filter(user)
        documents = {document.id: document for document in self.repository.list_documents(user.tenant_id)}
        for chunk in self.repository.list_chunks_for_tenant(user.tenant_id):
            document = documents.get(chunk.doc_id)
            if not document:
                continue
            if not access_filter.allows_document(document):
                continue
            if access_filter.allows_chunk(chunk):
                results.append((document, chunk))
        return results
