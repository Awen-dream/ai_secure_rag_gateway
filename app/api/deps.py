from functools import lru_cache

from app.core.config import settings
from app.domain.audit.services import AuditService
from app.domain.chat.services import ChatService
from app.domain.documents.services import DocumentService
from app.domain.prompts.services import PromptService
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.services import PolicyEngine
from app.infrastructure.db.repositories.sqlite import SQLiteRepository


@lru_cache
def get_repository() -> SQLiteRepository:
    return SQLiteRepository(settings.sqlite_path)


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(get_repository())


@lru_cache
def get_prompt_service() -> PromptService:
    return PromptService(get_repository())


@lru_cache
def get_policy_engine() -> PolicyEngine:
    return PolicyEngine(get_repository())


@lru_cache
def get_audit_service() -> AuditService:
    return AuditService(get_repository())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_document_service())


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        repository=get_repository(),
        retrieval_service=get_retrieval_service(),
        prompt_service=get_prompt_service(),
        policy_engine=get_policy_engine(),
        audit_service=get_audit_service(),
    )
