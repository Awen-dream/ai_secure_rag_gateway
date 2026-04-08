from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import (
    get_audit_service,
    get_keyword_backend,
    get_policy_engine,
    get_prompt_service,
    get_redis_client,
    get_retrieval_service,
    get_vector_backend,
)
from app.domain.auth.filter_builder import build_access_filter
from app.core.security import require_admin
from app.domain.auth.models import UserContext
from app.domain.audit.services import AuditService
from app.domain.prompts.models import PromptTemplate
from app.domain.prompts.services import PromptService
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
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.vectorstore.pgvector import PGVectorStore

router = APIRouter()


@router.get("/prompts", response_model=list[PromptTemplate])
def list_prompt_templates(
    _: UserContext = Depends(require_admin),
    service: PromptService = Depends(get_prompt_service),
) -> list[PromptTemplate]:
    return service.list_templates()


@router.post("/prompts", response_model=PromptTemplate)
def create_prompt_template(
    payload: PromptTemplate,
    _: UserContext = Depends(require_admin),
    service: PromptService = Depends(get_prompt_service),
) -> PromptTemplate:
    return service.add_template(payload)


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
    return [log.dict() for log in service.list_logs()]


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
