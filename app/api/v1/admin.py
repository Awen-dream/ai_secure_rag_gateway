from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.application.access.service import build_access_filter
from app.application.context.builder import ContextBuilderService
from app.application.prompting.builder import PromptBuilderService
from app.api.deps import (
    get_audit_service,
    get_context_builder_service,
    get_document_task_queue,
    get_feishu_source_sync_service,
    get_keyword_backend,
    get_policy_engine,
    get_prompt_builder_service,
    get_prompt_template_service,
    get_redis_client,
    get_retrieval_service,
    get_vector_backend,
)
from app.core.security import require_admin
from app.domain.auth.models import UserContext
from app.domain.audit.services import AuditService
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
    _: UserContext = Depends(require_admin),
    service: AuditService = Depends(get_audit_service),
) -> list[dict]:
    return [log.model_dump() for log in service.list_logs()]


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
) -> RetrievalBackendPlan:
    """Return backend-specific integration artifacts for debugging and deployment review."""

    if backend == "elasticsearch":
        terms = query.split()
        access_filter = build_access_filter(user)
        return RetrievalBackendPlan(
            backend="elasticsearch",
            execute_enabled=keyword_backend.can_execute(),
            artifacts={
                "access_filter": access_filter.model_dump(),
                "mapping": keyword_backend.build_index_mapping(),
                "search_body": keyword_backend.build_search_body(
                    query=query,
                    access_filter=access_filter,
                    terms=terms,
                    top_k=top_k,
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
                "ddl": vector_backend.build_table_ddl(),
                "upsert_sql": vector_backend.build_upsert_sql(),
                "search_sql": vector_backend.build_search_sql(access_filter, top_k),
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
