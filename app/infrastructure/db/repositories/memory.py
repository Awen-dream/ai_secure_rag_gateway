from __future__ import annotations

from typing import Dict, List

from app.domain.audit.models import AuditLog
from app.domain.chat.models import ChatMessage, ChatSession
from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.prompts.models import PromptTemplate
from app.domain.risk.models import PolicyDefinition


class InMemoryStore:
    def __init__(self) -> None:
        self.documents: Dict[str, DocumentRecord] = {}
        self.document_chunks: Dict[str, List[DocumentChunk]] = {}
        self.content_hashes: Dict[str, str] = {}
        self.chat_sessions: Dict[str, ChatSession] = {}
        self.chat_messages: Dict[str, List[ChatMessage]] = {}
        self.prompt_templates: Dict[str, List[PromptTemplate]] = {}
        self.policies: Dict[str, PolicyDefinition] = {}
        self.audit_logs: List[AuditLog] = []


store = InMemoryStore()
