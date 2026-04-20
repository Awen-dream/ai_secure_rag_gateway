from __future__ import annotations

import uuid
import time
from datetime import datetime

from app.domain.auth.models import UserContext
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.domain.sources.models import SourceSyncJob, SourceSyncRun
from app.domain.sources.schemas import (
    FeishuBatchSyncItemResponse,
    FeishuBatchSyncRequest,
    FeishuBatchSyncResponse,
    FeishuImportRequest,
    FeishuImportResponse,
    FeishuListSourceItemResponse,
    FeishuListSourcesRequest,
    FeishuListSourcesResponse,
    FeishuRunJobsResponse,
    FeishuSyncJobResponse,
    FeishuSyncJobUpsertRequest,
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
        source_sync_workflow=None,
    ) -> None:
        self.feishu_client = feishu_client
        self.repository = repository
        self.document_service = document_service
        self.task_queue = task_queue
        self.ingestion_orchestrator = ingestion_orchestrator
        self.source_sync_workflow = source_sync_workflow

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
        self.document_service.touch_source_last_seen_system(document.id)
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

    def sync_sources(
        self,
        payload: FeishuBatchSyncRequest,
        user: UserContext,
        *,
        mode: str = "manual",
        job_id: str | None = None,
    ) -> FeishuBatchSyncResponse:
        """Synchronize multiple Feishu sources and return item-level outcomes plus aggregate stats."""

        summary, _ = self._sync_sources_internal(payload, user, mode=mode, job_id=job_id)
        return summary

    def _sync_sources_internal(
        self,
        payload: FeishuBatchSyncRequest,
        user: UserContext,
        *,
        mode: str = "manual",
        job_id: str | None = None,
    ) -> tuple[FeishuBatchSyncResponse, list[str]]:
        """Internal sync helper that also returns listed source-document identifiers for reconciliation."""

        items_to_sync, listed_count, next_cursor, listed_source_document_ids = self._resolve_sync_items(payload)
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
            run_id=None,
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
        run_id = f"sync_run_{uuid.uuid4().hex[:12]}"
        self.repository.append_source_sync_run(
            SourceSyncRun(
                id=run_id,
                tenant_id=user.tenant_id,
                provider=self.feishu_client.provider,
                triggered_by=user.user_id,
                mode=mode,
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
                    "job_id": job_id,
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
        summary.run_id = run_id
        return summary, listed_source_document_ids

    def list_sync_runs(self, user: UserContext) -> list[SourceSyncRun]:
        """Return persisted sync runs for the current tenant and connector."""

        return self.repository.list_source_sync_runs(user.tenant_id, self.feishu_client.provider)

    def upsert_sync_job(self, payload: FeishuSyncJobUpsertRequest, user: UserContext) -> FeishuSyncJobResponse:
        """Create or update one Feishu sync job and persist its cursor for resumable runs."""

        if not payload.source_root and not payload.space_id:
            raise ValueError("Feishu sync job requires source_root or space_id.")

        existing = None
        if payload.job_id:
            existing = self.repository.get_source_sync_job(user.tenant_id, self.feishu_client.provider, payload.job_id)

        now = utcnow()
        job = SourceSyncJob(
            id=existing.id if existing else (payload.job_id or f"sync_job_{uuid.uuid4().hex[:12]}"),
            tenant_id=user.tenant_id,
            provider=self.feishu_client.provider,
            name=payload.name,
            created_by=existing.created_by if existing else user.user_id,
            source_root=payload.source_root,
            space_id=payload.space_id,
            parent_node_token=payload.parent_node_token,
            cursor=payload.cursor if payload.cursor is not None else (existing.cursor if existing else None),
            limit=payload.limit,
            continue_on_error=payload.continue_on_error,
            default_owner_id=payload.default_owner_id,
            default_department_scope=list(payload.default_department_scope),
            default_visibility_scope=list(payload.default_visibility_scope),
            default_security_level=payload.default_security_level,
            default_tags=list(payload.default_tags),
            default_async_mode=payload.default_async_mode,
            enabled=payload.enabled,
            status=("disabled" if not payload.enabled else (existing.status if existing else "idle")),
            last_error=existing.last_error if existing else None,
            run_count=existing.run_count if existing else 0,
            success_count=existing.success_count if existing else 0,
            failure_count=existing.failure_count if existing else 0,
            managed_source_document_ids=list(existing.managed_source_document_ids) if existing else [],
            cycle_seen_source_document_ids=list(existing.cycle_seen_source_document_ids) if existing else [],
            last_run_id=existing.last_run_id if existing else None,
            last_run_status=existing.last_run_status if existing else None,
            last_run_at=existing.last_run_at if existing else None,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self.repository.save_source_sync_job(job)
        return self._to_sync_job_response(job)

    def list_sync_jobs(self, user: UserContext) -> list[FeishuSyncJobResponse]:
        """Return persisted Feishu sync jobs for the current tenant."""

        return [
            self._to_sync_job_response(job)
            for job in self.repository.list_source_sync_jobs(user.tenant_id, self.feishu_client.provider)
        ]

    def run_sync_job(self, job_id: str, user: UserContext) -> FeishuBatchSyncResponse:
        """Execute one saved sync job using its persisted cursor and update the checkpoint."""

        if self.source_sync_workflow:
            workflow = self.source_sync_workflow.build_run_sync_job_workflow(self._run_sync_job_native)
            if workflow is not None:
                state = workflow.invoke({"job_id": job_id, "user": user})
                summary = state.get("summary") if isinstance(state, dict) else None
                if isinstance(summary, FeishuBatchSyncResponse):
                    return summary
        return self._run_sync_job_native(job_id, user)

    def _run_sync_job_native(self, job_id: str, user: UserContext) -> FeishuBatchSyncResponse:
        """Native implementation of one saved sync job execution."""

        job = self.repository.get_source_sync_job(user.tenant_id, self.feishu_client.provider, job_id)
        if not job:
            raise KeyError(job_id)
        if not job.enabled:
            raise ValueError(f"Feishu sync job {job_id} is disabled.")

        if not job.cursor:
            job.cycle_seen_source_document_ids = []
        job.status = "running"
        job.last_error = None
        job.updated_at = utcnow()
        self.repository.save_source_sync_job(job)

        try:
            summary, listed_source_document_ids = self._sync_sources_internal(
                FeishuBatchSyncRequest(
                    source_root=job.source_root,
                    space_id=job.space_id,
                    parent_node_token=job.parent_node_token,
                    cursor=job.cursor,
                    limit=job.limit,
                    continue_on_error=job.continue_on_error,
                    default_owner_id=job.default_owner_id,
                    default_department_scope=list(job.default_department_scope),
                    default_visibility_scope=list(job.default_visibility_scope),
                    default_security_level=job.default_security_level,
                    default_tags=list(job.default_tags),
                    default_async_mode=job.default_async_mode,
                ),
                user,
                mode="job",
                job_id=job.id,
            )
        except Exception as exc:
            job.status = "failed"
            job.last_error = str(exc)
            job.run_count += 1
            job.failure_count += 1
            job.last_run_status = "failed"
            job.last_run_at = utcnow()
            job.updated_at = job.last_run_at
            self.repository.save_source_sync_job(job)
            raise

        resolved_status = self._resolve_run_status(summary)
        merged_cycle_ids = list(
            dict.fromkeys([*job.cycle_seen_source_document_ids, *[item for item in listed_source_document_ids if item]])
        )
        job.cursor = summary.next_cursor
        job.status = self._resolve_job_status(job.enabled, resolved_status)
        job.last_error = None if resolved_status != "failed" else "sync job failed"
        job.run_count += 1
        if resolved_status in {"success", "partial_success", "empty"}:
            job.success_count += 1
        else:
            job.failure_count += 1
        job.last_run_id = summary.run_id
        job.last_run_status = resolved_status
        if summary.next_cursor is None:
            stale_source_ids = sorted(set(job.managed_source_document_ids) - set(merged_cycle_ids))
            self._retire_missing_documents(user.tenant_id, stale_source_ids)
            job.managed_source_document_ids = merged_cycle_ids
            job.cycle_seen_source_document_ids = []
        else:
            job.cycle_seen_source_document_ids = merged_cycle_ids
        job.last_run_at = utcnow()
        job.updated_at = job.last_run_at
        self.repository.save_source_sync_job(job)
        return summary

    def run_enabled_sync_jobs(self, user: UserContext) -> FeishuRunJobsResponse:
        """Run every enabled Feishu sync job once so external schedulers have a single entrypoint."""

        items: list[FeishuBatchSyncResponse] = []
        failed_jobs = 0
        skipped_jobs = 0
        jobs = self.repository.list_source_sync_jobs(user.tenant_id, self.feishu_client.provider)
        for job in jobs:
            if not job.enabled:
                skipped_jobs += 1
                continue
            if job.status == "running":
                skipped_jobs += 1
                continue
            try:
                items.append(self.run_sync_job(job.id, user))
            except Exception:
                failed_jobs += 1

        return FeishuRunJobsResponse(
            total_jobs=len(jobs),
            succeeded_jobs=len(items),
            failed_jobs=failed_jobs,
            skipped_jobs=skipped_jobs,
            items=items,
        )

    def run_enabled_sync_jobs_all_tenants(self) -> FeishuRunJobsResponse:
        """Run every enabled Feishu sync job across tenants for scheduler workers."""

        items: list[FeishuBatchSyncResponse] = []
        failed_jobs = 0
        skipped_jobs = 0
        jobs = self.repository.list_source_sync_jobs_all(self.feishu_client.provider)
        for job in jobs:
            if not job.enabled:
                skipped_jobs += 1
                continue
            if job.status == "running":
                skipped_jobs += 1
                continue
            scheduler_user = self._build_scheduler_user(job)
            try:
                items.append(self.run_sync_job(job.id, scheduler_user))
            except Exception:
                failed_jobs += 1

        return FeishuRunJobsResponse(
            total_jobs=len(jobs),
            succeeded_jobs=len(items),
            failed_jobs=failed_jobs,
            skipped_jobs=skipped_jobs,
            items=items,
        )

    def run_scheduler_forever(self, poll_seconds: float) -> None:
        """Run the Feishu sync scheduler loop forever using a fixed polling interval."""

        sleep_seconds = max(float(poll_seconds), 1.0)
        while True:
            self.run_enabled_sync_jobs_all_tenants()
            time.sleep(sleep_seconds)

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
    ) -> tuple[list[FeishuImportRequest], int, str | None, list[str]]:
        if payload.items:
            return list(payload.items), len(payload.items), None, []

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
        listed_source_document_ids = [item.external_document_id for item in page.items if item.external_document_id]
        return items, len(page.items), page.next_cursor, listed_source_document_ids

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

    @staticmethod
    def _to_sync_job_response(job: SourceSyncJob) -> FeishuSyncJobResponse:
        return FeishuSyncJobResponse(
            job_id=job.id,
            provider=job.provider,
            name=job.name,
            source_root=job.source_root,
            space_id=job.space_id,
            parent_node_token=job.parent_node_token,
            cursor=job.cursor,
            limit=job.limit,
            continue_on_error=job.continue_on_error,
            default_owner_id=job.default_owner_id,
            default_department_scope=list(job.default_department_scope),
            default_visibility_scope=list(job.default_visibility_scope),
            default_security_level=job.default_security_level,
            default_tags=list(job.default_tags),
            default_async_mode=job.default_async_mode,
            enabled=job.enabled,
            status=job.status,
            last_error=job.last_error,
            run_count=job.run_count,
            success_count=job.success_count,
            failure_count=job.failure_count,
            last_run_id=job.last_run_id,
            last_run_status=job.last_run_status,
            last_run_at=job.last_run_at.isoformat() if job.last_run_at else None,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
        )

    @staticmethod
    def _resolve_job_status(enabled: bool, run_status: str) -> str:
        if not enabled:
            return "disabled"
        if run_status == "failed":
            return "failed"
        return "idle"

    def _retire_missing_documents(self, tenant_id: str, stale_source_ids: list[str]) -> None:
        """Retire current documents whose external source ids disappeared from a completed sync cycle."""

        if not stale_source_ids:
            return
        stale_set = set(stale_source_ids)
        for document in self.repository.list_documents(tenant_id):
            if document.source_connector != self.feishu_client.provider:
                continue
            if not document.current:
                continue
            if not document.source_document_id or document.source_document_id not in stale_set:
                continue
            self.document_service.retire_document_system(
                document.id,
                reason="retired because source disappeared from Feishu sync scope",
            )

    @staticmethod
    def _build_scheduler_user(job: SourceSyncJob) -> UserContext:
        """Build an internal admin-like user context for scheduler-driven job execution."""

        return UserContext(
            user_id=job.created_by or "system_scheduler",
            tenant_id=job.tenant_id,
            department_id=(job.default_department_scope[0] if job.default_department_scope else "system"),
            role="admin",
            clearance_level=10,
        )
