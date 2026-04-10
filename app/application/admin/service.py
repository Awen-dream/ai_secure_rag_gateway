from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from app.application.evaluation.service import OfflineEvaluationService
from app.domain.audit.services import AuditService
from app.domain.documents.models import DocumentLifecycleStatus, DocumentRecord, DocumentStatus
from app.domain.documents.services import DocumentService
from app.domain.evaluation.models import EvalSample
from app.domain.prompts.template_service import PromptTemplateService
from app.domain.risk.services import PolicyEngine
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore


def utcnow() -> datetime:
    return datetime.utcnow()


class AdminConsoleService:
    """Aggregate admin-facing data for dashboard, dataset management and document operations."""

    def __init__(
        self,
        repository: MetadataRepository,
        document_service: DocumentService,
        audit_service: AuditService,
        evaluation_service: OfflineEvaluationService,
        eval_dataset_store: LocalEvalDatasetStore,
        prompt_template_service: PromptTemplateService,
        policy_engine: PolicyEngine,
        redis_client: RedisClient,
        task_queue: DocumentIngestionTaskQueue,
        feishu_source_service: FeishuSourceSyncService,
    ) -> None:
        self.repository = repository
        self.document_service = document_service
        self.audit_service = audit_service
        self.evaluation_service = evaluation_service
        self.eval_dataset_store = eval_dataset_store
        self.prompt_template_service = prompt_template_service
        self.policy_engine = policy_engine
        self.redis_client = redis_client
        self.task_queue = task_queue
        self.feishu_source_service = feishu_source_service

    def build_dashboard_summary(self) -> dict:
        """Return one aggregated operations dashboard payload for the admin console."""

        documents = self.repository.list_documents()
        audit_logs = self.repository.list_audit_logs()
        retrieval_metrics = self.audit_service.retrieval_metrics().model_dump()
        risk_metrics = self.audit_service.risk_metrics()
        prompts = self.prompt_template_service.list_templates()
        policies = self.policy_engine.list_policies()
        dataset = self.evaluation_service.list_samples()
        runs = self.evaluation_service.list_runs(limit=10)
        release_readiness = self.evaluation_service.build_release_readiness_report()
        shadow_report = self.evaluation_service.build_shadow_report()
        now = utcnow()
        recent_cutoff = now - timedelta(hours=24)

        status_counts = Counter(_document_status_value(document) for document in documents)
        lifecycle_counts = Counter(_document_lifecycle_value(document) for document in documents)
        source_counts = Counter(document.source_type for document in documents)
        connector_counts = Counter(document.source_connector or "manual" for document in documents)
        security_counts = Counter(str(document.security_level) for document in documents)
        question_counts = Counter(log.query for log in audit_logs if log.query.strip())
        cited_document_counts = Counter()
        unique_users = set()
        active_sessions = set()
        recent_queries = 0
        for log in audit_logs:
            unique_users.add(log.user_id)
            active_sessions.add(log.session_id)
            if log.created_at >= recent_cutoff:
                recent_queries += 1
            for item in log.retrieval_docs_json:
                title = item.get("title") or item.get("doc_id") or "unknown"
                cited_document_counts[title] += 1

        latest_offline_run = next((run for run in runs if run.mode == "offline"), None)
        latest_shadow_run = next((run for run in runs if run.mode == "shadow"), None)
        queue_health = self.task_queue.health()
        feishu_health = self.feishu_source_service.health_check()

        return {
            "generated_at": now.isoformat(),
            "documents": {
                "total": len(documents),
                "current": sum(1 for document in documents if document.current),
                "successful": status_counts.get(DocumentStatus.SUCCESS.value, 0),
                "pending": status_counts.get(DocumentStatus.PENDING.value, 0),
                "failed": status_counts.get(DocumentStatus.FAILED.value, 0),
                "retired": status_counts.get(DocumentStatus.RETIRED.value, 0),
                "by_status": dict(status_counts),
                "active": lifecycle_counts.get(DocumentLifecycleStatus.ACTIVE.value, 0),
                "deprecated": lifecycle_counts.get(DocumentLifecycleStatus.DEPRECATED.value, 0),
                "lifecycle_retired": lifecycle_counts.get(DocumentLifecycleStatus.RETIRED.value, 0),
                "by_lifecycle_status": dict(lifecycle_counts),
                "stale": len(self.list_stale_documents()),
                "by_source_type": dict(source_counts),
                "by_connector": dict(connector_counts),
                "by_security_level": dict(security_counts),
            },
            "traffic": {
                "total_queries": len(audit_logs),
                "recent_queries_24h": recent_queries,
                "unique_users": len(unique_users),
                "active_sessions": len(active_sessions),
            },
            "quality": {
                **retrieval_metrics,
                "latest_release_decision": release_readiness.decision,
                "latest_shadow_winner": shadow_report.winner,
            },
            "risk": risk_metrics,
            "evaluation": {
                "dataset_size": len(dataset),
                "latest_offline_run_id": latest_offline_run.run_id if latest_offline_run else "",
                "latest_shadow_run_id": latest_shadow_run.run_id if latest_shadow_run else "",
                "release_readiness": release_readiness.model_dump(mode="json"),
                "shadow_report": shadow_report.model_dump(mode="json"),
            },
            "operations": {
                "prompt_templates": {
                    "total": len(prompts),
                    "enabled": sum(1 for template in prompts if template.enabled),
                    "scenes": sorted({template.scene for template in prompts}),
                },
                "policies": {
                    "total": len(policies),
                    "enabled": sum(1 for policy in policies if policy.enabled),
                },
                "cache": {
                    "execute_enabled": self.redis_client.can_execute(),
                    "reachable": self.redis_client.ping(),
                },
                "queue": queue_health,
                "feishu": feishu_health,
            },
            "top_questions": [
                {"query": query, "count": count}
                for query, count in question_counts.most_common(10)
            ],
            "top_documents": [
                {"title": title, "hits": hits}
                for title, hits in cited_document_counts.most_common(10)
            ],
        }

    def list_documents(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        lifecycle_status: Optional[str] = None,
        source_type: Optional[str] = None,
        search: Optional[str] = None,
        current_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        """Return admin-facing document inventory records with chunk counts and filters."""

        documents = self.repository.list_documents(tenant_id)
        normalized_search = (search or "").strip().lower()
        filtered: list[DocumentRecord] = []
        for document in documents:
            if status and _document_status_value(document) != status:
                continue
            if lifecycle_status and _document_lifecycle_value(document) != lifecycle_status:
                continue
            if source_type and document.source_type != source_type:
                continue
            if current_only and not document.current:
                continue
            if normalized_search and normalized_search not in f"{document.title} {' '.join(document.tags)}".lower():
                continue
            filtered.append(document)

        filtered.sort(key=lambda item: item.updated_at, reverse=True)
        result: list[dict] = []
        for document in filtered[:limit]:
            chunks = self.repository.list_chunks_for_document(document.id)
            result.append(
                {
                    **document.model_dump(mode="json"),
                    "chunk_count": len(chunks),
                    "is_indexed": bool(chunks) and document.status == DocumentStatus.SUCCESS,
                }
            )
        return result

    def retire_document(self, doc_id: str, reason: str = "retired by admin console") -> dict:
        """Retire one document system-wide for admin remediation workflows."""

        document = self.document_service.retire_document_system(doc_id, reason=reason)
        return document.model_dump(mode="json")

    def deprecate_document(self, doc_id: str, reason: str = "deprecated by admin console") -> dict:
        """Mark one document deprecated without deleting its history."""

        document = self.document_service.deprecate_document_system(doc_id, reason=reason)
        return document.model_dump(mode="json")

    def replace_document(self, doc_id: str, replaced_by_doc_id: str, reason: str) -> dict:
        """Link one document to its replacement and deprecate the old snapshot."""

        document = self.document_service.replace_document_system(doc_id, replaced_by_doc_id, reason=reason)
        return document.model_dump(mode="json")

    def restore_document(
        self,
        doc_id: str,
        reason: str = "restored by admin console",
        source_last_seen_at: Optional[datetime] = None,
    ) -> dict:
        """Restore one document to active lifecycle state."""

        document = self.document_service.restore_document_system(
            doc_id,
            reason=reason,
            source_last_seen_at=source_last_seen_at,
        )
        return document.model_dump(mode="json")

    def list_stale_documents(self, tenant_id: Optional[str] = None, threshold_days: int = 30) -> list[dict]:
        """Return stale documents whose source_last_seen_at is older than the threshold."""

        documents = self.document_service.list_stale_documents_system(tenant_id, threshold_days=threshold_days)
        return [
            {
                **document.model_dump(mode="json"),
                "chunk_count": len(self.repository.list_chunks_for_document(document.id)),
            }
            for document in documents
            if document.lifecycle_status in {DocumentLifecycleStatus.ACTIVE, DocumentLifecycleStatus.DEPRECATED}
        ]

    def replace_evaluation_dataset(self, samples: list[EvalSample]) -> dict:
        """Replace the evaluation dataset with the provided samples."""

        count = self.eval_dataset_store.replace_samples(samples)
        return {"sample_count": count, "dataset_path": str(self.eval_dataset_store.dataset_path)}

    def evaluation_dataset_overview(self) -> dict:
        """Return the current dataset inventory plus baseline snapshot for admin workflows."""

        return {
            "stats": self.evaluation_service.dataset_stats().model_dump(mode="json"),
            "baseline": self.evaluation_service.get_quality_baseline().model_dump(mode="json"),
            "sample_count": len(self.evaluation_service.list_samples()),
        }

    def bootstrap_evaluation_dataset(self, limit: int = 20) -> dict:
        """Generate a starter evaluation dataset from current successful documents."""

        documents = [
            document
            for document in self.repository.list_documents()
            if document.current and document.status == DocumentStatus.SUCCESS
        ]
        documents.sort(key=lambda item: item.updated_at, reverse=True)
        samples: list[EvalSample] = []
        for index, document in enumerate(documents[:limit], start=1):
            chunks = self.repository.list_chunks_for_document(document.id)
            expected_terms = []
            if chunks:
                snippet = chunks[0].text.strip().replace("\n", " ")
                if snippet:
                    expected_terms.append(snippet[:24])
            samples.append(
                EvalSample(
                    id=f"bootstrap_{index}_{document.id}",
                    query=f"{document.title} 的核心内容是什么？",
                    scene="standard_qa",
                    expected_doc_ids=[document.id],
                    expected_titles=[document.title],
                    expected_answer_contains=expected_terms[:1],
                    tenant_id=document.tenant_id,
                    department_id=document.department_scope[0] if document.department_scope else "engineering",
                    role="admin",
                    clearance_level=max(document.security_level, 1),
                    tags=document.tags,
                    metadata={
                        "document_id": document.id,
                        "source_type": document.source_type,
                        "section_name": chunks[0].section_name if chunks else "",
                    },
                )
            )

        count = self.eval_dataset_store.replace_samples(samples)
        return {
            "sample_count": count,
            "dataset_path": str(self.eval_dataset_store.dataset_path),
            "sample_ids": [sample.id for sample in samples[:10]],
        }


def _document_status_value(document: DocumentRecord) -> str:
    return document.status.value if isinstance(document.status, DocumentStatus) else str(document.status)


def _document_lifecycle_value(document: DocumentRecord) -> str:
    if isinstance(document.lifecycle_status, DocumentLifecycleStatus):
        return document.lifecycle_status.value
    return str(document.lifecycle_status)
