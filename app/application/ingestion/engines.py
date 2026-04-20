from __future__ import annotations

import uuid
from datetime import datetime
from typing import Protocol

from app.application.ingestion.document_parser import extract_text_from_bytes
from app.application.ingestion.pipelines import chunk_document
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.documents.models import DocumentChunk, DocumentLifecycleStatus, DocumentRecord, DocumentStatus
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentIngestionEngine(Protocol):
    """Execution engine contract for staged document ingestion."""

    engine_name: str

    def process_document(
        self,
        *,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
        retrieval_cache: RetrievalCache | None,
        doc_id: str,
    ) -> DocumentRecord:
        ...


class NativeDocumentIngestionEngine:
    """Default ingestion engine that preserves the current in-process parser/chunker/indexer flow."""

    engine_name = "native"

    def process_document(
        self,
        *,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
        retrieval_cache: RetrievalCache | None,
        doc_id: str,
    ) -> DocumentRecord:
        document = repository.get_document(doc_id)
        if not document:
            raise KeyError(doc_id)

        try:
            source_bytes = source_store.read_source(document.id, document.source_type)
            self._update_document_status(repository, document, DocumentStatus.PARSING)

            content = extract_text_from_bytes(source_bytes, document.source_type, document.title)
            self._update_document_status(repository, document, DocumentStatus.CHUNKING)

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

            self._update_document_status(repository, document, DocumentStatus.EMBEDDING)
            self._update_document_status(repository, document, DocumentStatus.INDEXING)

            previous_records = self._resolve_previous_records(repository, document)

            document.status = DocumentStatus.SUCCESS
            document.last_error = None
            document.current = True
            document.updated_at = utcnow()
            repository.save_document(document, chunks, [item.id for item in previous_records])
            self._mark_replaced_versions(repository, document, previous_records)
            indexing_service.upsert_document(document, chunks)
            if retrieval_cache:
                retrieval_cache.invalidate_all()
            return document
        except Exception as exc:
            document.status = DocumentStatus.FAILED
            document.last_error = str(exc)
            document.updated_at = utcnow()
            repository.update_document(document)
            return document

    @staticmethod
    def _update_document_status(
        repository: MetadataRepository,
        document: DocumentRecord,
        status: DocumentStatus,
    ) -> None:
        document.status = status
        document.last_error = None
        document.updated_at = utcnow()
        repository.update_document(document)

    @staticmethod
    def _resolve_previous_records(
        repository: MetadataRepository,
        document: DocumentRecord,
    ) -> list[DocumentRecord]:
        if document.source_connector and document.source_document_id:
            history = repository.list_documents_by_source_ref(
                document.tenant_id,
                document.source_connector,
                document.source_document_id,
            )
        else:
            history = repository.list_documents_by_title(document.tenant_id, document.title)

        return [record for record in history if record.id != document.id and record.current]

    @staticmethod
    def _mark_replaced_versions(
        repository: MetadataRepository,
        document: DocumentRecord,
        previous_records: list[DocumentRecord],
    ) -> None:
        for previous in previous_records:
            previous.current = False
            previous.lifecycle_status = DocumentLifecycleStatus.DEPRECATED
            previous.replaced_by_doc_id = document.id
            previous.lifecycle_reason = f"superseded by {document.id}"
            previous.updated_at = utcnow()
            repository.update_document(previous)
