from functools import lru_cache

from app.application.admin.service import AdminConsoleService
from app.application.chat.orchestrator import ChatOrchestrator
from app.application.context.builder import ContextBuilderService
from app.application.evaluation.engines import EvaluationExecutionEngine, NativeEvaluationExecutionEngine
from app.application.evaluation.service import OfflineEvaluationService
from app.application.generation.service import GenerationService
from app.application.ingestion.engines import DocumentIngestionEngine, NativeDocumentIngestionEngine
from app.application.retrieval.llm_reranker import LLMReranker
from app.application.prompting.builder import PromptBuilderService
from app.application.retrieval.rerank import RetrievalRerankService
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.application.retrieval.planning import RecallPlanningService
from app.application.query.planning import QueryPlanningService
from app.application.session.cache import SessionCache
from app.application.session.service import SessionContextService
from app.application.query.understanding import QueryUnderstandingService
from app.core.config import settings
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.audit.services import AuditService
from app.domain.documents.services import DocumentService
from app.domain.prompts.template_service import PromptTemplateService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.retrieval.rerankers import CompositeReranker, HeuristicReranker, RetrievalReranker
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.output_guard import OutputGuard
from app.domain.risk.rate_limit import RateLimitService
from app.domain.risk.services import PolicyEngine
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.external_sources.feishu import FeishuClient
from app.infrastructure.frameworks.llamaindex_eval import LlamaIndexEvaluationExecutionEngine
from app.infrastructure.frameworks.llamaindex_ingestion import LlamaIndexDocumentIngestionEngine
from app.infrastructure.llm.deepseek_client import DeepSeekClient
from app.infrastructure.llm.openai_client import OpenAIClient
from app.infrastructure.llm.openai_embeddings import OpenAIEmbeddingClient
from app.infrastructure.llm.qwen_client import QwenClient
from app.infrastructure.llm.router import LLMRouter
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue, DocumentIngestionWorker
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore
from app.infrastructure.storage.local_eval_baseline_store import LocalEvalBaselineStore
from app.infrastructure.storage.local_eval_run_store import LocalEvalRunStore
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore


@lru_cache
def get_repository() -> MetadataRepository:
    """Return the configured metadata repository used by services in the current process."""

    if settings.repository_backend == "postgres":
        from app.infrastructure.db.repositories.postgres import PostgresRepository

        if not settings.postgres_dsn:
            raise ValueError("APP_POSTGRES_DSN is required when APP_REPOSITORY_BACKEND=postgres")
        return PostgresRepository(
            dsn=settings.postgres_dsn,
            auto_init_schema=settings.postgres_auto_init_schema,
        )
    return SQLiteRepository(settings.sqlite_path)


@lru_cache
def get_redis_client() -> RedisClient:
    """Return the configured Redis cache client used for session cache, retrieval cache and rate limiting."""

    return RedisClient(mode=settings.redis_mode, url=settings.redis_url)


@lru_cache
def get_session_cache() -> SessionCache:
    """Return the session cache service backed by Redis or its local fallback."""

    return SessionCache(
        redis_client=get_redis_client(),
        ttl_seconds=settings.session_cache_ttl_seconds,
    )


@lru_cache
def get_retrieval_cache() -> RetrievalCache:
    """Return the retrieval result cache service backed by Redis or its local fallback."""

    return RetrievalCache(
        redis_client=get_redis_client(),
        ttl_seconds=settings.retrieval_cache_ttl_seconds,
    )


@lru_cache
def get_rate_limit_service() -> RateLimitService:
    """Return the fixed-window rate limiter used by user-facing chat endpoints."""

    return RateLimitService(
        redis_client=get_redis_client(),
        window_seconds=settings.rate_limit_window_seconds,
        max_requests=settings.rate_limit_max_requests,
    )


@lru_cache
def get_keyword_backend() -> ElasticsearchSearch:
    """Return the keyword retrieval backend adapter."""

    return ElasticsearchSearch(
        index_name=settings.elasticsearch_index,
        mode=settings.elasticsearch_mode,
        endpoint=settings.elasticsearch_endpoint,
        auto_init_index=settings.elasticsearch_auto_init_index,
    )


@lru_cache
def get_vector_backend() -> PGVectorStore:
    """Return the vector retrieval backend adapter."""

    return PGVectorStore(
        table_name=settings.pgvector_table,
        embedding_dimension=settings.embedding_dimension,
        mode=settings.pgvector_mode,
        dsn=settings.pgvector_dsn,
        auto_init_schema=settings.pgvector_auto_init_schema,
        embedding_client=get_embedding_client(),
    )


@lru_cache
def get_indexing_service() -> RetrievalIndexingService:
    """Return the service responsible for syncing retrieval backends after document changes."""

    return RetrievalIndexingService(
        keyword_backend=get_keyword_backend(),
        vector_backend=get_vector_backend(),
    )


@lru_cache
def get_document_source_store() -> LocalDocumentSourceStore:
    """Return the local staging store that keeps raw uploads available for background ingestion."""

    return LocalDocumentSourceStore(settings.document_staging_dir)


@lru_cache
def get_eval_dataset_store() -> LocalEvalDatasetStore:
    """Return the local evaluation dataset store used by offline evaluation tooling."""

    return LocalEvalDatasetStore(settings.eval_dataset_path)


@lru_cache
def get_eval_run_store() -> LocalEvalRunStore:
    """Return the local evaluation run store used by offline/shadow evaluation history."""

    return LocalEvalRunStore(settings.eval_runs_dir)


@lru_cache
def get_eval_baseline_store() -> LocalEvalBaselineStore:
    """Return the local evaluation quality-baseline store used by release gating."""

    return LocalEvalBaselineStore(settings.eval_baseline_path)


@lru_cache
def get_document_ingestion_orchestrator() -> DocumentIngestionOrchestrator:
    """Return the document ingestion orchestrator used by synchronous and background upload flows."""

    return DocumentIngestionOrchestrator(
        repository=get_repository(),
        indexing_service=get_indexing_service(),
        source_store=get_document_source_store(),
        retrieval_cache=get_retrieval_cache(),
        ingestion_engine=get_document_ingestion_engine(),
    )


@lru_cache
def get_document_ingestion_engine() -> DocumentIngestionEngine:
    """Return the configured document-ingestion execution engine."""

    if settings.ingestion_engine == "llamaindex":
        return LlamaIndexDocumentIngestionEngine()
    return NativeDocumentIngestionEngine()


@lru_cache
def get_document_task_queue() -> DocumentIngestionTaskQueue:
    """Return the document ingestion task queue used by upload and retry endpoints."""

    return DocumentIngestionTaskQueue(
        redis_client=get_redis_client(),
        queue_name=settings.document_ingestion_queue_name,
    )


@lru_cache
def get_document_ingestion_worker() -> DocumentIngestionWorker:
    """Return the dedicated ingestion worker wiring used by local scripts and tests."""

    return DocumentIngestionWorker(
        task_queue=get_document_task_queue(),
        orchestrator=get_document_ingestion_orchestrator(),
        poll_seconds=settings.document_ingestion_worker_poll_seconds,
    )


@lru_cache
def get_document_service() -> DocumentService:
    """Return the document domain service with repository, staging and orchestration dependencies wired."""

    return DocumentService(
        repository=get_repository(),
        indexing_service=get_indexing_service(),
        source_store=get_document_source_store(),
        ingestion_orchestrator=get_document_ingestion_orchestrator(),
    )


@lru_cache
def get_prompt_template_service() -> PromptTemplateService:
    """Return the prompt template registry and validation service."""

    return PromptTemplateService(get_repository())


@lru_cache
def get_prompt_builder_service() -> PromptBuilderService:
    """Return the prompt build service used to render template + context payloads."""

    return PromptBuilderService(get_prompt_template_service())


@lru_cache
def get_policy_engine() -> PolicyEngine:
    """Return the risk policy engine."""

    return PolicyEngine(get_repository())


@lru_cache
def get_output_guard() -> OutputGuard:
    """Return the output-side safety guard used before answers leave the gateway."""

    return OutputGuard()


@lru_cache
def get_context_builder_service() -> ContextBuilderService:
    """Return the context assembly service used by chat and prompt preview flows."""

    return ContextBuilderService()


@lru_cache
def get_generation_service() -> GenerationService:
    """Return the generation service that owns model invocation, guard, and validation."""

    return GenerationService(
        prompt_template_service=get_prompt_template_service(),
        output_guard=get_output_guard(),
        llm_client=get_llm_router().get_client("generation"),
    )


@lru_cache
def get_offline_evaluation_service() -> OfflineEvaluationService:
    """Return the offline evaluation service used by admin evaluation endpoints."""

    return OfflineEvaluationService(
        dataset_store=get_eval_dataset_store(),
        baseline_store=get_eval_baseline_store(),
        run_store=get_eval_run_store(),
        retrieval_service=get_retrieval_service(),
        context_builder=get_context_builder_service(),
        prompt_builder=get_prompt_builder_service(),
        generation_service=get_generation_service(),
        execution_engine=get_evaluation_execution_engine(),
    )


@lru_cache
def get_evaluation_execution_engine() -> EvaluationExecutionEngine:
    """Return the configured evaluation execution engine."""

    if settings.evaluation_engine == "llamaindex":
        return LlamaIndexEvaluationExecutionEngine()
    return NativeEvaluationExecutionEngine()


@lru_cache
def get_admin_console_service() -> AdminConsoleService:
    """Return the admin console service that aggregates dashboard, document and evaluation management data."""

    return AdminConsoleService(
        repository=get_repository(),
        document_service=get_document_service(),
        audit_service=get_audit_service(),
        evaluation_service=get_offline_evaluation_service(),
        eval_dataset_store=get_eval_dataset_store(),
        prompt_template_service=get_prompt_template_service(),
        policy_engine=get_policy_engine(),
        redis_client=get_redis_client(),
        task_queue=get_document_task_queue(),
        feishu_source_service=get_feishu_source_sync_service(),
    )


@lru_cache
def get_audit_service() -> AuditService:
    """Return the audit service used for metrics and trace logging."""

    return AuditService(get_repository())


@lru_cache
def get_openai_client() -> OpenAIClient:
    """Return the OpenAI Responses API adapter used by the chat service."""

    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
        max_output_tokens=settings.openai_max_output_tokens,
        temperature=settings.openai_temperature,
    )


def get_qwen_client() -> QwenClient:
    """Return the Qwen client wired against Alibaba Cloud's OpenAI-compatible chat API."""

    return QwenClient(
        api_key=settings.qwen_api_key,
        model=settings.qwen_model,
        base_url=settings.qwen_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
        max_output_tokens=settings.qwen_max_output_tokens,
        temperature=settings.qwen_temperature,
    )


def get_deepseek_client() -> DeepSeekClient:
    """Return the DeepSeek client wired against its OpenAI-compatible chat API."""

    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.deepseek_timeout_seconds,
        max_output_tokens=settings.deepseek_max_output_tokens,
        temperature=settings.deepseek_temperature,
    )


def get_llm_router() -> LLMRouter:
    """Return the per-purpose LLM router used by generation, rerank and understanding."""

    return LLMRouter.build(
        default_provider=settings.llm_default_provider,
        generation_provider=settings.llm_generation_provider or settings.llm_default_provider,
        query_understanding_provider=(
            settings.llm_query_understanding_provider or settings.llm_default_provider
        ),
        reranker_provider=settings.llm_reranker_provider or settings.llm_default_provider,
        clients=[
            get_openai_client(),
            get_qwen_client(),
            get_deepseek_client(),
        ],
    )


@lru_cache
def get_embedding_client() -> OpenAIEmbeddingClient:
    """Return the embedding client used by vector indexing and semantic retrieval."""

    return OpenAIEmbeddingClient(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        timeout_seconds=settings.embedding_timeout_seconds,
        dimensions=settings.embedding_dimensions,
        enabled=settings.embedding_provider == "openai",
    )


@lru_cache
def get_retrieval_reranker() -> RetrievalReranker:
    """Return the configured reranker implementation used inside the retrieval rerank layer."""

    heuristic = HeuristicReranker(
        mode="cross-encoder-fallback" if settings.reranker_mode != "disabled" else "disabled",
        top_n=settings.reranker_top_n,
    )
    if settings.reranker_mode == "disabled":
        return heuristic
    if settings.reranker_mode == "llm":
        return CompositeReranker(
            primary=LLMReranker(client=get_llm_router().get_client("reranker"), top_n=settings.reranker_top_n),
            fallback=heuristic,
        )
    if settings.reranker_mode == "heuristic":
        return HeuristicReranker(mode="heuristic", top_n=settings.reranker_top_n)
    return heuristic


@lru_cache
def get_query_understanding_service() -> QueryUnderstandingService:
    """Return the shared query-understanding service used by planning and retrieval."""

    return QueryUnderstandingService(get_llm_router().get_client("query_understanding"))


@lru_cache
def get_query_planning_service() -> QueryPlanningService:
    """Return the query-planning service that composes understanding and rewrite planning."""

    return QueryPlanningService(get_query_understanding_service())


@lru_cache
def get_recall_planning_service() -> RecallPlanningService:
    """Return the recall-planning service that maps query intent into retrieval execution plans."""

    return RecallPlanningService()


@lru_cache
def get_retrieval_rerank_service() -> RetrievalRerankService:
    """Return the rerank-layer service used after multi-backend retrieval execution."""

    return RetrievalRerankService(get_retrieval_reranker())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """Return the hybrid retrieval service backed by Elasticsearch and PGVector adapters."""

    return RetrievalService(
        document_service=get_document_service(),
        keyword_backend=get_keyword_backend(),
        vector_backend=get_vector_backend(),
        retrieval_cache=get_retrieval_cache(),
        reranker=get_retrieval_reranker(),
        query_planning=get_query_planning_service(),
        recall_planning=get_recall_planning_service(),
        rerank_service=get_retrieval_rerank_service(),
    )


@lru_cache
def get_chat_service() -> ChatOrchestrator:
    """Return the application chat orchestrator with all online-QA dependencies wired."""

    return ChatOrchestrator(
        repository=get_repository(),
        retrieval_service=get_retrieval_service(),
        policy_engine=get_policy_engine(),
        audit_service=get_audit_service(),
        prompt_builder=get_prompt_builder_service(),
        generation_service=get_generation_service(),
        context_builder=get_context_builder_service(),
        session_cache=get_session_cache(),
        session_context_service=SessionContextService(
            get_repository(),
            query_planning=get_query_planning_service(),
        ),
    )


@lru_cache
def get_feishu_client() -> FeishuClient:
    """Return the Feishu connector used by admin-triggered external source imports."""

    return FeishuClient(
        base_url=settings.feishu_base_url,
        app_id=settings.feishu_app_id,
        app_secret=settings.feishu_app_secret,
    )


@lru_cache
def get_feishu_source_sync_service() -> FeishuSourceSyncService:
    """Return the Feishu external source import service."""

    return FeishuSourceSyncService(
        feishu_client=get_feishu_client(),
        repository=get_repository(),
        document_service=get_document_service(),
        task_queue=get_document_task_queue(),
        ingestion_orchestrator=get_document_ingestion_orchestrator(),
    )
