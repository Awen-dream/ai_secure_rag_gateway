from __future__ import annotations

import uuid
from datetime import datetime

from app.application.ingestion.document_parser import extract_text_from_bytes
from app.application.ingestion.pipelines import chunk_document
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentIngestionOrchestrator:
    """Runs the staged ingestion workflow from raw source bytes through retrieval indexing."""

    def __init__(
        self,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
    ) -> None:
        self.repository = repository
        self.indexing_service = indexing_service
        self.source_store = source_store

    def process_document(self, doc_id: str) -> DocumentRecord:
        """Advance one document through parsing, chunking, embedding and indexing."""

        document = self.repository.get_document(doc_id)
        if not document:
            raise KeyError(doc_id)

        try:
            source_bytes = self.source_store.read_source(document.id, document.source_type)
            self._update_document_status(document, DocumentStatus.PARSING)

            content = extract_text_from_bytes(source_bytes, document.source_type, document.title)
            self._update_document_status(document, DocumentStatus.CHUNKING)

            chunk_payloads = chunk_document(content, source_type=document.source_type)
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

            self._update_document_status(document, DocumentStatus.EMBEDDING)
            self._update_document_status(document, DocumentStatus.INDEXING)

            previous_ids = self._resolve_previous_ids(document)

            document.status = DocumentStatus.SUCCESS
            document.last_error = None
            document.current = True
            document.updated_at = utcnow()
            self.repository.save_document(document, chunks, previous_ids)
            self.indexing_service.upsert_document(document, chunks)
            return document
        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.last_error = str(exc)
            document.updated_at = utcnow()
            self.repository.update_document(document)
            return document

    def retry_document(self, doc_id: str) -> DocumentRecord:
        """Reset one failed or pending document back to pending before reprocessing."""

        document = self.repository.get_document(doc_id)
        if not document:
            raise KeyError(doc_id)
        document.status = DocumentStatus.PENDING
        document.last_error = None
        document.updated_at = utcnow()
        self.repository.update_document(document)
        return document

    def _update_document_status(self, document: DocumentRecord, status: DocumentStatus) -> None:
        document.status = status
        document.last_error = None
        document.updated_at = utcnow()
        self.repository.update_document(document)

    def _resolve_previous_ids(self, document: DocumentRecord) -> list[str]:
        if document.source_connector and document.source_document_id:
            history = self.repository.list_documents_by_source_ref(
                document.tenant_id,
                document.source_connector,
                document.source_document_id,
            )
        else:
            history = self.repository.list_documents_by_title(document.tenant_id, document.title)

        return [record.id for record in history if record.id != document.id and record.current]
