from __future__ import annotations

import uuid
from datetime import datetime

from app.domain.auth.models import UserContext
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.domain.sources.models import SourceSyncRun
from app.domain.sources.schemas import (
    FeishuBatchSyncItemResponse,
    FeishuBatchSyncRequest,
    FeishuBatchSyncResponse,
    FeishuImportRequest,
    FeishuImportResponse,
    FeishuListSourceItemResponse,
    FeishuListSourcesRequest,
    FeishuListSourcesResponse,
    SourceSyncAction,
)
from app.infrastructure.external_sources.base import ExternalSourceConnector, ExternalSourceItem
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue


def utcnow() -> datetime:
    return datetime.utcnow()


class FeishuSourceSyncService:
    """Imports Feishu docs and wiki-backed docs into the gateway document pipeline."""

    def __init__(
        self,
        feishu_client: ExternalSourceConnector,
        repository: MetadataRepository,
        document_service: DocumentService,
        task_queue: DocumentIngestionTaskQueue,
        ingestion_orchestrator: DocumentIngestionOrchestrator,
    ) -> None:
        self.feishu_client = feishu_client
        self.repository = repository
        self.document_service = document_service
        self.task_queue = task_queue
        self.ingestion_orchestrator = ingestion_orchestrator

    def import_source(self, payload: FeishuImportRequest, user: UserContext) -> FeishuImportResponse:
        """Fetch one Feishu source, register it as a gateway document, and optionally enqueue ingestion."""

        document_content = self.feishu_client.fetch_document(payload.source)
        source_reference = self.feishu_client.parse_source(payload.source)
        previous_current = self.repository.find_current_document_by_source_ref(
            user.tenant_id,
            document_content.connector,
            document_content.external_document_id,
        )
        document = self.document_service.upload_document(
            DocumentUploadRequest(
                title=payload.title or document_content.title,
                content=document_content.content,
                source_type=document_content.source_type,
                source_uri=document_content.source_uri,
                source_connector=document_content.connector,
                source_document_id=document_content.external_document_id,
                source_document_version=document_content.external_version,
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
        sync_action = self._resolve_sync_action(previous_current, document)
        return FeishuImportResponse(
            source=payload.source,
            source_kind=source_reference.source_kind,
            document_id=document.id,
            document_status=document.status.value,
            document_version=document.version,
            sync_action=sync_action,
            queued=task_receipt is not None,
            task_id=task_receipt["task_id"] if task_receipt else None,
        )

    def sync_sources(self, payload: FeishuBatchSyncRequest, user: UserContext) -> FeishuBatchSyncResponse:
        """Synchronize multiple Feishu sources and return item-level outcomes plus aggregate stats."""

        items_to_sync, listed_count, next_cursor = self._resolve_sync_items(payload)
        results: list[FeishuBatchSyncItemResponse] = []
        for item in items_to_sync:
            try:
                response = self.import_source(item, user)
                results.append(
                    FeishuBatchSyncItemResponse(
                        source=response.source,
                        source_kind=response.source_kind,
                        success=True,
                        sync_action=response.sync_action,
                        document_id=response.document_id,
                        document_status=response.document_status,
                        document_version=response.document_version,
                        queued=response.queued,
                        task_id=response.task_id,
                    )
                )
            except Exception as exc:
                results.append(
                    FeishuBatchSyncItemResponse(
                        source=item.source,
                        success=False,
                        sync_action=SourceSyncAction.FAILED,
                        error=str(exc),
                    )
                )
                if not payload.continue_on_error:
                    break

        summary = FeishuBatchSyncResponse(
            total=len(results),
            listed_count=listed_count,
            succeeded=sum(1 for item in results if item.success),
            failed=sum(1 for item in results if not item.success),
            imported_new=sum(1 for item in results if item.sync_action == SourceSyncAction.IMPORTED_NEW),
            reused_current=sum(1 for item in results if item.sync_action == SourceSyncAction.REUSED_CURRENT),
            created_new_version=sum(1 for item in results if item.sync_action == SourceSyncAction.CREATED_NEW_VERSION),
            queued=sum(1 for item in results if item.queued),
            next_cursor=next_cursor,
            items=results,
        )
        self.repository.append_source_sync_run(
            SourceSyncRun(
                id=f"sync_run_{uuid.uuid4().hex[:12]}",
                tenant_id=user.tenant_id,
                provider=self.feishu_client.provider,
                triggered_by=user.user_id,
                mode="manual",
                continue_on_error=payload.continue_on_error,
                request_json={
                    "sources": [item.source for item in items_to_sync],
                    "item_count": len(items_to_sync),
                    "cursor": payload.cursor,
                    "limit": payload.limit,
                    "listing_mode": not payload.items,
                    "source_root": payload.source_root,
                    "space_id": payload.space_id,
                    "parent_node_token": payload.parent_node_token,
                },
                result_items_json=[item.model_dump() for item in results],
                total=summary.total,
                succeeded=summary.succeeded,
                failed=summary.failed,
                imported_new=summary.imported_new,
                reused_current=summary.reused_current,
                created_new_version=summary.created_new_version,
                queued=summary.queued,
                status=self._resolve_run_status(summary),
                created_at=utcnow(),
            )
        )
        return summary

    def list_sync_runs(self, user: UserContext) -> list[SourceSyncRun]:
        """Return persisted sync runs for the current tenant and connector."""

        return self.repository.list_source_sync_runs(user.tenant_id, self.feishu_client.provider)

    def list_sources(self, payload: FeishuListSourcesRequest) -> FeishuListSourcesResponse:
        """List Feishu spaces or nodes for admin exploration and sync preparation."""

        page = self.feishu_client.list_sources(
            cursor=payload.cursor,
            limit=payload.limit,
            source_root=payload.source_root,
            space_id=payload.space_id,
            parent_node_token=payload.parent_node_token,
        )
        return FeishuListSourcesResponse(
            listed_count=len(page.items),
            next_cursor=page.next_cursor,
            items=[self._to_list_item_response(item) for item in page.items],
        )

    def health_check(self) -> dict:
        """Return Feishu client health for admin diagnostics."""

        return self.feishu_client.health_check()

    @staticmethod
    def _resolve_sync_action(previous_current, document) -> str:
        if previous_current is None:
            return SourceSyncAction.IMPORTED_NEW
        if previous_current.id == document.id:
            return SourceSyncAction.REUSED_CURRENT
        return SourceSyncAction.CREATED_NEW_VERSION

    @staticmethod
    def _resolve_run_status(summary: FeishuBatchSyncResponse) -> str:
        if summary.total == 0:
            return "empty"
        if summary.failed == 0:
            return "success"
        if summary.succeeded == 0:
            return "failed"
        return "partial_success"

    def _resolve_sync_items(
        self,
        payload: FeishuBatchSyncRequest,
    ) -> tuple[list[FeishuImportRequest], int, str | None]:
        if payload.items:
            return list(payload.items), len(payload.items), None

        if not payload.source_root and not payload.space_id:
            raise ValueError("Feishu sync listing requires source_root or space_id when items are not provided.")
        page = self.feishu_client.list_sources(
            cursor=payload.cursor,
            limit=payload.limit,
            source_root=payload.source_root,
            space_id=payload.space_id,
            parent_node_token=payload.parent_node_token,
        )
        items = [self._build_listed_import_request(item, payload) for item in page.items]
        return items, len(page.items), page.next_cursor

    @staticmethod
    def _build_listed_import_request(
        item: ExternalSourceItem,
        payload: FeishuBatchSyncRequest,
    ) -> FeishuImportRequest:
        return FeishuImportRequest(
            source=item.source,
            title=item.title,
            owner_id=payload.default_owner_id,
            department_scope=list(payload.default_department_scope),
            visibility_scope=list(payload.default_visibility_scope),
            security_level=payload.default_security_level,
            tags=list(payload.default_tags),
            async_mode=payload.default_async_mode,
        )

    @staticmethod
    def _to_list_item_response(item: ExternalSourceItem) -> FeishuListSourceItemResponse:
        return FeishuListSourceItemResponse(
            source=item.source,
            source_kind=item.source_kind,
            external_document_id=item.external_document_id,
            title=item.title,
            space_id=item.space_id,
            node_token=item.node_token,
            parent_node_token=item.parent_node_token,
            obj_type=item.obj_type,
            has_child=item.has_child,
        )
