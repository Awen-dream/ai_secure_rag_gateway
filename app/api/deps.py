from functools import lru_cache

from app.core.config import settings
from app.domain.audit.services import AuditService
from app.domain.chat.services import ChatService
from app.domain.documents.services import DocumentService
from app.domain.prompts.services import PromptService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.services import PolicyEngine
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.vectorstore.pgvector import PGVectorStore


@lru_cache
def get_repository() -> SQLiteRepository:
    """Return the shared metadata repository used by services in the current process."""

    return SQLiteRepository(settings.sqlite_path)


@lru_cache
def get_keyword_backend() -> ElasticsearchSearch:
    """Return the keyword retrieval backend adapter."""

    return ElasticsearchSearch(index_name=settings.elasticsearch_index)


@lru_cache
def get_vector_backend() -> PGVectorStore:
    """Return the vector retrieval backend adapter."""

    return PGVectorStore(table_name=settings.pgvector_table)


@lru_cache
def get_indexing_service() -> RetrievalIndexingService:
    """Return the service responsible for syncing retrieval backends after document changes."""

    return RetrievalIndexingService(
        keyword_backend=get_keyword_backend(),
        vector_backend=get_vector_backend(),
    )


@lru_cache
def get_document_service() -> DocumentService:
    """Return the document domain service with repository and index sync dependencies wired."""

    return DocumentService(get_repository(), get_indexing_service())


@lru_cache
def get_prompt_service() -> PromptService:
    """Return the prompt template service."""

    return PromptService(get_repository())


@lru_cache
def get_policy_engine() -> PolicyEngine:
    """Return the risk policy engine."""

    return PolicyEngine(get_repository())


@lru_cache
def get_audit_service() -> AuditService:
    """Return the audit service used for metrics and trace logging."""

    return AuditService(get_repository())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """Return the hybrid retrieval service backed by Elasticsearch and PGVector adapters."""

    return RetrievalService(
        document_service=get_document_service(),
        keyword_backend=get_keyword_backend(),
        vector_backend=get_vector_backend(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Return the chat service with retrieval, policy and audit dependencies wired."""

    return ChatService(
        repository=get_repository(),
        retrieval_service=get_retrieval_service(),
        prompt_service=get_prompt_service(),
        policy_engine=get_policy_engine(),
        audit_service=get_audit_service(),
    )
