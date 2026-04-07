from __future__ import annotations

from typing import Optional, Protocol

from app.domain.audit.models import AuditLog
from app.domain.chat.models import ChatMessage, ChatSession
from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.prompts.models import PromptTemplate
from app.domain.risk.models import PolicyDefinition


class MetadataRepository(Protocol):
    """Shared persistence contract used by domain services regardless of the backing database."""

    def find_document_by_content_hash(self, tenant_id: str, content_hash: str) -> Optional[DocumentRecord]:
        ...

    def list_documents_by_title(self, tenant_id: str, title: str) -> list[DocumentRecord]:
        ...

    def save_document(self, document: DocumentRecord, chunks: list[DocumentChunk], previous_ids: list[str]) -> None:
        ...

    def update_document(self, document: DocumentRecord) -> None:
        ...

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        ...

    def list_documents(self, tenant_id: Optional[str] = None) -> list[DocumentRecord]:
        ...

    def list_chunks_for_document(self, doc_id: str) -> list[DocumentChunk]:
        ...

    def list_chunks_for_tenant(self, tenant_id: str) -> list[DocumentChunk]:
        ...

    def save_session(self, session: ChatSession) -> None:
        ...

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        ...

    def list_sessions(self, tenant_id: str, user_id: str) -> list[ChatSession]:
        ...

    def append_message(self, message: ChatMessage) -> None:
        ...

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        ...

    def save_prompt_template(self, template: PromptTemplate) -> None:
        ...

    def list_prompt_templates(self, scene: Optional[str] = None) -> list[PromptTemplate]:
        ...

    def save_policy(self, policy: PolicyDefinition) -> None:
        ...

    def list_policies(self) -> list[PolicyDefinition]:
        ...

    def append_audit_log(self, log: AuditLog) -> None:
        ...

    def list_audit_logs(self) -> list[AuditLog]:
        ...
