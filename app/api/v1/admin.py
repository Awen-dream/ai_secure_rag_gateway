from __future__ import annotations

from datetime import datetime
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.application.admin.service import AdminConsoleService
from app.application.access.service import build_access_filter
from app.application.context.builder import ContextBuilderService
from app.application.evaluation.service import OfflineEvaluationService
from app.application.prompting.builder import PromptBuilderService
from app.application.query.planning import QueryPlanningService
from app.application.retrieval.planning import RecallPlanningService
from app.api.deps import (
    get_audit_service,
    get_admin_console_service,
    get_context_builder_service,
    get_document_task_queue,
    get_offline_evaluation_service,
    get_feishu_source_sync_service,
    get_keyword_backend,
    get_policy_engine,
    get_prompt_builder_service,
    get_query_planning_service,
    get_recall_planning_service,
    get_prompt_template_service,
    get_redis_client,
    get_retrieval_service,
    get_vector_backend,
)
from app.core.security import require_admin
from app.domain.auth.models import UserContext
from app.domain.audit.services import AuditService
from app.domain.documents.admin_schemas import (
    DocumentLifecycleUpdateRequest,
    DocumentReplaceRequest,
    DocumentRestoreRequest,
    DocumentStaleQueryResponse,
)
from app.domain.evaluation.models import (
    EvalBulkAnnotationRequest,
    EvalBulkAnnotationResult,
    EvalDatasetExport,
    EvalDatasetImportRequest,
    EvalDatasetImportResult,
    EvalRunListItem,
    EvalRunResult,
    EvalSample,
    EvalSampleTemplate,
    EvalDatasetStats,
    EvalQualityBaseline,
    EvalTrendSummary,
    ReleaseGateReport,
    ReleaseReadinessReport,
    ShadowEvalRunResult,
    ShadowReportSummary,
)
from app.domain.prompts.models import (
    PromptPreviewRequest,
    PromptPreviewResponse,
    PromptTemplate,
    PromptValidationRequest,
    PromptValidationResult,
)
from app.domain.prompts.template_service import PromptTemplateService
from app.domain.retrieval.models import (
    RetrievalBackendHealth,
    RetrievalBackendInfo,
    RetrievalBackendPlan,
    RetrievalExplainRequest,
    RetrievalExplainResponse,
)
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import PolicyDefinition
from app.domain.risk.services import PolicyEngine
from app.domain.sources.schemas import (
    FeishuBatchSyncRequest,
    FeishuBatchSyncResponse,
    FeishuImportRequest,
    FeishuImportResponse,
    FeishuListSourcesRequest,
    FeishuListSourcesResponse,
    FeishuRunJobsResponse,
    FeishuSyncJobResponse,
    FeishuSyncJobUpsertRequest,
)
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.vectorstore.pgvector import PGVectorStore

router = APIRouter()


@router.get("/evaluation/dataset", response_model=list[EvalSample])
def list_evaluation_dataset(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> list[EvalSample]:
    """List the currently loaded offline evaluation dataset."""

    return service.list_samples()


@router.get("/evaluation/dataset/template", response_model=EvalSampleTemplate)
def get_evaluation_sample_template(
    scene: str = Query("standard_qa"),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalSampleTemplate:
    """Return a starter evaluation sample template for dataset authoring."""

    return service.build_sample_template(scene=scene)


@router.get("/evaluation/dataset/export", response_model=EvalDatasetExport)
def export_evaluation_dataset(
    export_format: str = Query("json"),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalDatasetExport:
    """Export the current evaluation dataset as JSON payload plus JSONL text."""

    return service.export_samples(export_format=export_format)


@router.get("/evaluation/dataset/stats", response_model=EvalDatasetStats)
def get_evaluation_dataset_stats(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalDatasetStats:
    """Return dataset coverage and labeling stats for evaluation sample governance."""

    return service.dataset_stats()


@router.get("/evaluation/dataset/overview")
def get_evaluation_dataset_overview(
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Return dataset stats plus current baseline snapshot for admin workbench screens."""

    return service.evaluation_dataset_overview()


@router.get("/evaluation/dataset/{sample_id}", response_model=EvalSample)
def get_evaluation_sample(
    sample_id: str,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalSample:
    """Return one evaluation sample for manual review and editing."""

    sample = service.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation sample not found.")
    return sample


@router.post("/evaluation/dataset/upsert", response_model=EvalSample)
def upsert_evaluation_sample(
    payload: EvalSample,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalSample:
    """Create or update one evaluation sample."""

    return service.upsert_sample(payload)


@router.post("/evaluation/dataset/import", response_model=EvalDatasetImportResult)
def import_evaluation_dataset(
    payload: EvalDatasetImportRequest,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalDatasetImportResult:
    """Import evaluation samples using replace or upsert semantics."""

    return service.import_samples(payload)


@router.delete("/evaluation/dataset/{sample_id}")
def delete_evaluation_sample(
    sample_id: str,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> dict:
    """Delete one evaluation sample by id."""

    deleted = service.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation sample not found.")
    return {"deleted": True, "sample_id": sample_id}


@router.post("/evaluation/dataset/bulk-annotate", response_model=EvalBulkAnnotationResult)
def bulk_annotate_evaluation_samples(
    payload: EvalBulkAnnotationRequest,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalBulkAnnotationResult:
    """Apply one labeling or review update to multiple evaluation samples."""

    return service.bulk_annotate(payload)


@router.post("/evaluation/dataset/replace")
def replace_evaluation_dataset(
    samples: list[EvalSample],
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Replace the admin evaluation dataset used by offline and shadow evaluation runs."""

    return service.replace_evaluation_dataset(samples)


@router.post("/evaluation/dataset/bootstrap")
def bootstrap_evaluation_dataset(
    limit: int = Query(20, ge=1, le=200),
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Generate a starter evaluation dataset from the current successful knowledge documents."""

    return service.bootstrap_evaluation_dataset(limit=limit)


@router.get("/evaluation/baseline", response_model=EvalQualityBaseline)
def get_evaluation_baseline(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalQualityBaseline:
    """Return the persisted quality baseline used by quality gate and release gate logic."""

    return service.get_quality_baseline()


@router.post("/evaluation/baseline", response_model=EvalQualityBaseline)
def update_evaluation_baseline(
    payload: EvalQualityBaseline,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalQualityBaseline:
    """Persist one quality baseline update for future evaluation and release-gate decisions."""

    return service.update_quality_baseline(payload)


@router.post("/evaluation/run", response_model=EvalRunResult)
def run_offline_evaluation(
    limit: Optional[int] = Query(None, ge=1, le=500),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalRunResult:
    """Run offline evaluation against the current retrieval and generation stack."""

    return service.run(limit=limit)


@router.post("/evaluation/run-shadow", response_model=ShadowEvalRunResult)
def run_shadow_evaluation(
    limit: Optional[int] = Query(None, ge=1, le=500),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> ShadowEvalRunResult:
    """Run a shadow evaluation that compares the active retrieval stack against a heuristic baseline."""

    return service.run_shadow(limit=limit)


@router.get("/evaluation/runs", response_model=list[EvalRunListItem])
def list_evaluation_runs(
    limit: int = Query(20, ge=1, le=200),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> list[EvalRunListItem]:
    """List persisted offline and shadow evaluation runs for later inspection."""

    return service.list_runs(limit=limit)


@router.get("/evaluation/runs/{run_id}")
def get_evaluation_run(
    run_id: str,
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> dict:
    """Return one persisted evaluation run payload by id."""

    payload = service.get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")
    return payload


@router.get("/evaluation/trend", response_model=EvalTrendSummary)
def get_evaluation_trend(
    history_limit: int = Query(10, ge=2, le=100),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> EvalTrendSummary:
    """Return evaluation trend summary and regression alerts based on persisted offline runs."""

    return service.build_trend_summary(history_limit=history_limit)


@router.get("/evaluation/release-readiness", response_model=ReleaseReadinessReport)
def get_release_readiness_report(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> ReleaseReadinessReport:
    """Return the release-readiness decision based on latest offline and shadow evaluation runs."""

    return service.build_release_readiness_report()


@router.get("/evaluation/release-gate", response_model=ReleaseGateReport)
def get_release_gate_report(
    allow_review: bool = Query(False),
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> ReleaseGateReport:
    """Return an explicit release-gate checklist and pass/fail result for CI or manual review."""

    return service.build_release_gate_report(allow_review=allow_review)


@router.get("/evaluation/shadow-report", response_model=ShadowReportSummary)
def get_shadow_report(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> ShadowReportSummary:
    """Return the latest shadow comparison summary for primary versus baseline retrieval."""

    return service.build_shadow_report()


@router.get("/prompts", response_model=list[PromptTemplate])
def list_prompt_templates(
    scene: Optional[str] = None,
    _: UserContext = Depends(require_admin),
    service: PromptTemplateService = Depends(get_prompt_template_service),
) -> list[PromptTemplate]:
    return service.list_templates(scene)


@router.post("/prompts", response_model=PromptTemplate)
def create_prompt_template(
    payload: PromptTemplate,
    _: UserContext = Depends(require_admin),
    service: PromptTemplateService = Depends(get_prompt_template_service),
) -> PromptTemplate:
    return service.add_template(payload)


@router.post("/prompts/{template_id}/enable", response_model=PromptTemplate)
def set_prompt_template_enabled(
    template_id: str,
    enabled: bool = Query(...),
    _: UserContext = Depends(require_admin),
    service: PromptTemplateService = Depends(get_prompt_template_service),
) -> PromptTemplate:
    """Enable or disable one prompt template version."""

    try:
        return service.set_template_enabled(template_id, enabled)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt template not found.") from exc


@router.post("/prompts/preview", response_model=PromptPreviewResponse)
def preview_prompt_template(
    payload: PromptPreviewRequest,
    user: UserContext = Depends(require_admin),
    prompt_builder: PromptBuilderService = Depends(get_prompt_builder_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    context_builder: ContextBuilderService = Depends(get_context_builder_service),
) -> PromptPreviewResponse:
    """Render the active prompt with live retrieval evidence for admin preview."""

    retrieved = retrieval_service.retrieve(user, payload.query, top_k=payload.top_k)
    assembled_context = context_builder.build(retrieved)
    return prompt_builder.preview_chat_prompt(
        scene=payload.scene,
        query=payload.query,
        assembled_context=assembled_context,
        session_summary=payload.session_summary,
    )


@router.post("/prompts/validate", response_model=PromptValidationResult)
def validate_prompt_output(
    payload: PromptValidationRequest,
    _: UserContext = Depends(require_admin),
    service: PromptTemplateService = Depends(get_prompt_template_service),
) -> PromptValidationResult:
    """Validate one answer against the active prompt template output schema."""

    return service.validate_output(payload.scene, payload.answer)


@router.get("/policies", response_model=list[PolicyDefinition])
def list_policies(
    _: UserContext = Depends(require_admin),
    service: PolicyEngine = Depends(get_policy_engine),
) -> list[PolicyDefinition]:
    return service.list_policies()


@router.post("/policies", response_model=PolicyDefinition)
def create_policy(
    payload: PolicyDefinition,
    _: UserContext = Depends(require_admin),
    service: PolicyEngine = Depends(get_policy_engine),
) -> PolicyDefinition:
    return service.add_policy(payload)


@router.get("/audit")
def list_audit_logs(
    limit: int = Query(100, ge=1, le=500),
    risk_level: Optional[str] = None,
    action: Optional[str] = None,
    scene: Optional[str] = None,
    query: Optional[str] = None,
    _: UserContext = Depends(require_admin),
    service: AuditService = Depends(get_audit_service),
) -> list[dict]:
    logs = service.list_logs()
    filtered = []
    normalized_query = (query or "").strip().lower()
    for log in logs:
        if risk_level and log.risk_level != risk_level:
            continue
        if action and log.action != action:
            continue
        if scene and log.scene != scene:
            continue
        if normalized_query and normalized_query not in f"{log.query} {log.rewritten_query}".lower():
            continue
        filtered.append(log.model_dump())
        if len(filtered) >= limit:
            break
    return filtered


@router.get("/dashboard/summary")
def get_admin_dashboard_summary(
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Return aggregated operations, quality, risk and evaluation summaries for the admin dashboard."""

    return service.build_dashboard_summary()


@router.get("/documents")
def list_admin_documents(
    tenant_id: Optional[str] = None,
    status: Optional[str] = None,
    lifecycle_status: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    current_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> list[dict]:
    """Return admin document inventory with status filters and chunk-count metadata."""

    return service.list_documents(
        tenant_id=tenant_id,
        status=status,
        lifecycle_status=lifecycle_status,
        source_type=source_type,
        search=search,
        current_only=current_only,
        limit=limit,
    )


@router.post("/documents/{doc_id}/retire")
def retire_admin_document(
    doc_id: str,
    reason: str = Query("retired by admin console"),
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Retire one document version for admin remediation and index cleanup workflows."""

    try:
        return service.retire_document(doc_id, reason=reason)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc


@router.post("/documents/{doc_id}/deprecate")
def deprecate_admin_document(
    doc_id: str,
    payload: DocumentLifecycleUpdateRequest,
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Mark one document deprecated so it stops serving as active knowledge."""

    try:
        return service.deprecate_document(doc_id, reason=payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc


@router.post("/documents/{doc_id}/replace")
def replace_admin_document(
    doc_id: str,
    payload: DocumentReplaceRequest,
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Link one document to a newer replacement and deprecate the old one."""

    try:
        return service.replace_document(doc_id, payload.replaced_by_doc_id, reason=payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/documents/{doc_id}/restore")
def restore_admin_document(
    doc_id: str,
    payload: DocumentRestoreRequest,
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> dict:
    """Restore one deprecated or retired document back to active lifecycle state."""

    try:
        seen_at = datetime.fromisoformat(payload.source_last_seen_at) if payload.source_last_seen_at else None
        return service.restore_document(doc_id, reason=payload.reason, source_last_seen_at=seen_at)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/documents/stale", response_model=DocumentStaleQueryResponse)
def list_stale_admin_documents(
    tenant_id: Optional[str] = None,
    threshold_days: int = Query(30, ge=1, le=3650),
    _: UserContext = Depends(require_admin),
    service: AdminConsoleService = Depends(get_admin_console_service),
) -> DocumentStaleQueryResponse:
    """List documents that have not been seen from their external source within the threshold."""

    return DocumentStaleQueryResponse(
        threshold_days=threshold_days,
        documents=service.list_stale_documents(tenant_id=tenant_id, threshold_days=threshold_days),
    )


@router.get("/retrieval/backends", response_model=list[RetrievalBackendInfo])
def list_retrieval_backends(
    _: UserContext = Depends(require_admin),
    service: RetrievalService = Depends(get_retrieval_service),
) -> list[RetrievalBackendInfo]:
    """Return the active retrieval backends and their runtime configuration summary."""

    return service.backend_info()


@router.post("/retrieval/explain", response_model=RetrievalExplainResponse)
def explain_retrieval(
    payload: RetrievalExplainRequest,
    user: UserContext = Depends(require_admin),
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalExplainResponse:
    """Explain hybrid retrieval behavior for an admin-visible query under current access scope."""

    return service.explain(user, payload.query, payload.top_k)


@router.get("/retrieval/backends/{backend}/plan", response_model=RetrievalBackendPlan)
def get_retrieval_backend_plan(
    backend: str,
    query: str = Query("企业知识访问网关"),
    top_k: int = Query(5, ge=1, le=20),
    user: UserContext = Depends(require_admin),
    keyword_backend: ElasticsearchSearch = Depends(get_keyword_backend),
    vector_backend: PGVectorStore = Depends(get_vector_backend),
    query_planning: QueryPlanningService = Depends(get_query_planning_service),
    recall_planning: RecallPlanningService = Depends(get_recall_planning_service),
) -> RetrievalBackendPlan:
    """Return backend-specific integration artifacts for debugging and deployment review."""

    query_plan = query_planning.plan(query)
    recall_plan = recall_planning.plan(query_plan, top_k=top_k)

    if backend == "elasticsearch":
        access_filter = build_access_filter(user)
        return RetrievalBackendPlan(
            backend="elasticsearch",
            execute_enabled=keyword_backend.can_execute(),
            artifacts={
                "access_filter": access_filter.model_dump(),
                "query_plan": asdict(query_plan),
                "recall_plan": {
                    "keyword_query": recall_plan.keyword_query,
                    "keyword_terms": recall_plan.keyword_terms,
                    "exact_match_terms": recall_plan.exact_match_terms,
                    "tag_filters": recall_plan.filters.tag_filters,
                    "year_filters": recall_plan.filters.year_filters,
                    "candidate_pool": recall_plan.candidate_pool,
                },
                "mapping": keyword_backend.build_index_mapping(),
                "search_body": keyword_backend.build_search_body(
                    query=recall_plan.keyword_query,
                    access_filter=access_filter,
                    terms=recall_plan.keyword_terms,
                    top_k=top_k,
                    tag_filters=recall_plan.filters.tag_filters,
                    year_filters=recall_plan.filters.year_filters,
                    exact_terms=recall_plan.exact_match_terms,
                ),
            },
        )

    if backend == "pgvector":
        access_filter = build_access_filter(user)
        return RetrievalBackendPlan(
            backend="pgvector",
            execute_enabled=vector_backend.can_execute(),
            artifacts={
                "access_filter": access_filter.model_dump(),
                "query_plan": asdict(query_plan),
                "recall_plan": {
                    "vector_query": recall_plan.vector_query,
                    "tag_filters": recall_plan.filters.tag_filters,
                    "year_filters": recall_plan.filters.year_filters,
                    "candidate_pool": recall_plan.candidate_pool,
                },
                "ddl": vector_backend.build_table_ddl(),
                "upsert_sql": vector_backend.build_upsert_sql(),
                "search_sql": vector_backend.build_search_sql(
                    access_filter,
                    top_k,
                    tag_filters=recall_plan.filters.tag_filters,
                    year_filters=recall_plan.filters.year_filters,
                ),
            },
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unsupported retrieval backend.")


@router.post("/retrieval/backends/pgvector/init-schema")
def init_pgvector_schema(
    _: UserContext = Depends(require_admin),
    vector_backend: PGVectorStore = Depends(get_vector_backend),
) -> dict:
    """Initialize pgvector schema when PostgreSQL execution mode is configured."""

    return vector_backend.initialize_schema()


@router.post("/retrieval/backends/elasticsearch/init-index")
def init_elasticsearch_index(
    _: UserContext = Depends(require_admin),
    keyword_backend: ElasticsearchSearch = Depends(get_keyword_backend),
) -> dict:
    """Initialize the Elasticsearch index when remote execution mode is configured."""

    return keyword_backend.initialize_index()


@router.get("/retrieval/backends/{backend}/health", response_model=RetrievalBackendHealth)
def get_retrieval_backend_health(
    backend: str,
    _: UserContext = Depends(require_admin),
    keyword_backend: ElasticsearchSearch = Depends(get_keyword_backend),
    vector_backend: PGVectorStore = Depends(get_vector_backend),
) -> RetrievalBackendHealth:
    """Return backend reachability information for retrieval infrastructure diagnostics."""

    if backend == "elasticsearch":
        return keyword_backend.health_check()
    if backend == "pgvector":
        return vector_backend.health_check()
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unsupported retrieval backend.")


@router.get("/cache/health")
def get_cache_health(
    _: UserContext = Depends(require_admin),
    redis_client: RedisClient = Depends(get_redis_client),
) -> dict:
    """Return Redis cache availability information for cache, session and rate-limit diagnostics."""

    return {
        "backend": "redis",
        "execute_enabled": redis_client.can_execute(),
        "reachable": redis_client.ping(),
        "mode": redis_client.mode,
    }


@router.get("/queue/document-ingestion/health")
def get_document_ingestion_queue_health(
    _: UserContext = Depends(require_admin),
    task_queue: DocumentIngestionTaskQueue = Depends(get_document_task_queue),
) -> dict:
    """Return document-ingestion queue reachability and queue-depth information."""

    return task_queue.health()


@router.get("/sources/feishu/health")
def get_feishu_source_health(
    _: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> dict:
    """Return Feishu connector reachability and credential configuration status."""

    return service.health_check()


@router.post("/sources/feishu/list", response_model=FeishuListSourcesResponse)
def list_feishu_sources(
    payload: FeishuListSourcesRequest,
    _: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuListSourcesResponse:
    """List Feishu spaces or child wiki nodes for admin exploration and sync preparation."""

    return service.list_sources(payload)


@router.get("/sources/feishu/jobs", response_model=list[FeishuSyncJobResponse])
def list_feishu_sync_jobs(
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> list[FeishuSyncJobResponse]:
    """List saved Feishu sync jobs and their latest cursor checkpoints."""

    return service.list_sync_jobs(user)


@router.post("/sources/feishu/jobs", response_model=FeishuSyncJobResponse)
def upsert_feishu_sync_job(
    payload: FeishuSyncJobUpsertRequest,
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuSyncJobResponse:
    """Create or update one saved Feishu sync job."""

    return service.upsert_sync_job(payload, user)


@router.post("/sources/feishu/jobs/{job_id}/run", response_model=FeishuBatchSyncResponse)
def run_feishu_sync_job(
    job_id: str,
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuBatchSyncResponse:
    """Execute one saved Feishu sync job using its persisted cursor."""

    return service.run_sync_job(job_id, user)


@router.post("/sources/feishu/jobs/run-enabled", response_model=FeishuRunJobsResponse)
def run_enabled_feishu_sync_jobs(
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuRunJobsResponse:
    """Run every enabled Feishu sync job once for scheduler-style entrypoints."""

    return service.run_enabled_sync_jobs(user)


@router.post("/sources/feishu/import", response_model=FeishuImportResponse)
def import_feishu_source(
    payload: FeishuImportRequest,
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuImportResponse:
    """Import one Feishu docx or wiki-backed doc into the gateway document pipeline."""

    return service.import_source(payload, user)


@router.post("/sources/feishu/sync", response_model=FeishuBatchSyncResponse)
def sync_feishu_sources(
    payload: FeishuBatchSyncRequest,
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> FeishuBatchSyncResponse:
    """Synchronize multiple Feishu sources and return aggregate outcomes for admin workflows."""

    return service.sync_sources(payload, user)


@router.get("/sources/feishu/runs")
def list_feishu_sync_runs(
    user: UserContext = Depends(require_admin),
    service: FeishuSourceSyncService = Depends(get_feishu_source_sync_service),
) -> list[dict]:
    """Return persisted Feishu sync runs for admin diagnostics and operations."""

    return [run.model_dump() for run in service.list_sync_runs(user)]
