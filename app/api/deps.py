from functools import lru_cache

from app.application.conversation.memory import ConversationManager
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.application.query.understanding import QueryUnderstandingService
from app.core.config import settings
from app.application.conversation.session_cache import SessionCache
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.audit.services import AuditService
from app.domain.chat.services import ChatService
from app.domain.documents.services import DocumentService
from app.domain.prompts.services import PromptService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.retrieval.rerankers import HeuristicReranker
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.output_guard import OutputGuard
from app.domain.risk.rate_limit import RateLimitService
from app.domain.risk.services import PolicyEngine
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.external_sources.feishu import FeishuClient
from app.infrastructure.llm.openai_client import OpenAIClient
from app.infrastructure.llm.openai_embeddings import OpenAIEmbeddingClient
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue, DocumentIngestionWorker
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
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
def get_document_ingestion_orchestrator() -> DocumentIngestionOrchestrator:
    """Return the document ingestion orchestrator used by synchronous and background upload flows."""

    return DocumentIngestionOrchestrator(
        repository=get_repository(),
        indexing_service=get_indexing_service(),
        source_store=get_document_source_store(),
    )


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
def get_prompt_service() -> PromptService:
    """Return the prompt template service."""

    return PromptService(get_repository())


@lru_cache
def get_policy_engine() -> PolicyEngine:
    """Return the risk policy engine."""

    return PolicyEngine(get_repository())


@lru_cache
def get_output_guard() -> OutputGuard:
    """Return the output-side safety guard used before answers leave the gateway."""

    return OutputGuard()


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
def get_retrieval_reranker() -> HeuristicReranker:
    """Return the active reranker used after first-pass hybrid retrieval fusion."""

    return HeuristicReranker(mode=settings.reranker_mode, top_n=settings.reranker_top_n)


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """Return the hybrid retrieval service backed by Elasticsearch and PGVector adapters."""

    return RetrievalService(
        document_service=get_document_service(),
        keyword_backend=get_keyword_backend(),
        vector_backend=get_vector_backend(),
        retrieval_cache=get_retrieval_cache(),
        reranker=get_retrieval_reranker(),
        query_understanding=QueryUnderstandingService(get_openai_client()),
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Return the chat service with retrieval, policy and audit dependencies wired."""

    return ChatService(
        repository=get_repository(),
        retrieval_service=get_retrieval_service(),
        prompt_service=get_prompt_service(),
        policy_engine=get_policy_engine(),
        output_guard=get_output_guard(),
        audit_service=get_audit_service(),
        openai_client=get_openai_client(),
        session_cache=get_session_cache(),
        conversation_manager=ConversationManager(get_repository()),
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
