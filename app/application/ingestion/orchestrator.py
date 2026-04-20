from __future__ import annotations

from datetime import datetime

from app.application.ingestion.engines import DocumentIngestionEngine, NativeDocumentIngestionEngine
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.documents.models import DocumentRecord, DocumentStatus
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
        retrieval_cache: RetrievalCache | None = None,
        ingestion_engine: DocumentIngestionEngine | None = None,
    ) -> None:
        self.repository = repository
        self.indexing_service = indexing_service
        self.source_store = source_store
        self.retrieval_cache = retrieval_cache
        self.ingestion_engine = ingestion_engine or NativeDocumentIngestionEngine()

    def process_document(self, doc_id: str) -> DocumentRecord:
        """Advance one document through parsing, chunking, embedding and indexing."""

        return self.ingestion_engine.process_document(
            repository=self.repository,
            indexing_service=self.indexing_service,
            source_store=self.source_store,
            retrieval_cache=self.retrieval_cache,
            doc_id=doc_id,
        )

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
