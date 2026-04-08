from __future__ import annotations

from app.domain.auth.models import UserContext
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.domain.sources.schemas import FeishuImportRequest, FeishuImportResponse
from app.infrastructure.external_sources.feishu import FeishuClient
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue


class FeishuSourceSyncService:
    """Imports Feishu docs and wiki-backed docs into the gateway document pipeline."""

    def __init__(
        self,
        feishu_client: FeishuClient,
        document_service: DocumentService,
        task_queue: DocumentIngestionTaskQueue,
        ingestion_orchestrator: DocumentIngestionOrchestrator,
    ) -> None:
        self.feishu_client = feishu_client
        self.document_service = document_service
        self.task_queue = task_queue
        self.ingestion_orchestrator = ingestion_orchestrator

    def import_source(self, payload: FeishuImportRequest, user: UserContext) -> FeishuImportResponse:
        """Fetch one Feishu source, register it as a gateway document, and optionally enqueue ingestion."""

        document_content = self.feishu_client.fetch_document(payload.source)
        source_reference = self.feishu_client.parse_source(payload.source)
        document = self.document_service.upload_document(
            DocumentUploadRequest(
                title=payload.title or document_content.title,
                content=document_content.content,
                source_type=document_content.source_type,
                source_uri=document_content.source_uri,
                owner_id=payload.owner_id,
                department_scope=payload.department_scope,
                visibility_scope=payload.visibility_scope,
                security_level=payload.security_level,
                tags=list(dict.fromkeys([*payload.tags, "feishu", source_reference.source_kind])),
                async_mode=payload.async_mode,
            ),
            user,
        )
        if payload.async_mode:
            task_receipt = self.task_queue.enqueue_document(document.id)
        else:
            document = self.ingestion_orchestrator.process_document(document.id)
            task_receipt = None
        return FeishuImportResponse(
            source=payload.source,
            source_kind=source_reference.source_kind,
            document_id=document.id,
            document_status=document.status.value,
            queued=task_receipt is not None,
            task_id=task_receipt["task_id"] if task_receipt else None,
        )

    def health_check(self) -> dict:
        """Return Feishu client health for admin diagnostics."""

        return self.feishu_client.health_check()
