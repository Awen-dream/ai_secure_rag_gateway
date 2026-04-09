from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.application.access.service import build_access_signature
from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.application.session.cache import SessionCache
from app.application.session.service import SessionContextService
from app.application.session.summarizer import summarize_long_history
from app.domain.audit.services import AuditService
from app.domain.auth.models import UserContext
from app.domain.chat.models import ChatMessage, ChatSession, SessionStatus
from app.domain.chat.schemas import ChatQueryRequest, ChatQueryResponse
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.services import PolicyEngine
from app.infrastructure.db.repositories.base import MetadataRepository


def utcnow() -> datetime:
    return datetime.utcnow()


class ChatOrchestrator:
    """Application-layer orchestration for the secure RAG chat flow."""

    def __init__(
        self,
        repository: MetadataRepository,
        retrieval_service: RetrievalService,
        policy_engine: PolicyEngine,
        audit_service: AuditService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
        context_builder: ContextBuilderService | None = None,
        session_cache: SessionCache | None = None,
        session_context_service: SessionContextService | None = None,
    ) -> None:
        self.repository = repository
        self.retrieval_service = retrieval_service
        self.policy_engine = policy_engine
        self.audit_service = audit_service
        self.prompt_builder = prompt_builder
        self.generation_service = generation_service
        self.context_builder = context_builder or ContextBuilderService()
        self.session_cache = session_cache
        self.session_context_service = session_context_service or SessionContextService(repository)

    def query(self, payload: ChatQueryRequest, user: UserContext) -> ChatQueryResponse:
        """Execute one secure RAG chat request end to end."""

        started_at = utcnow()
        session = self._get_or_create_session(payload, user)
        if self.session_cache and not session.summary:
            cached_summary = self.session_cache.get_summary(session.id)
            if cached_summary:
                session.summary = cached_summary
        session_context = self.session_context_service.build_context(session, user, payload.query)
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        retrieved = self.retrieval_service.retrieve(
            user,
            payload.query,
            query_plan=session_context.query_plan,
        )
        risk_action, risk_level = self.policy_engine.evaluate(user, session_context.rewritten_query, len(retrieved))
        input_risk_action = risk_action
        assembled_context = self.context_builder.build(retrieved)
        prompt_build = self.prompt_builder.build_chat_prompt(
            scene=payload.scene,
            query=session_context.rewritten_query,
            assembled_context=assembled_context,
            session_summary=session_context.session_summary,
        )
        generation = self.generation_service.generate_chat_answer(
            user=user,
            prompt_build=prompt_build,
            input_risk_action=risk_action,
            input_risk_level=risk_level,
        )
        citations = assembled_context.citations
        answer = generation.answer
        risk_action = generation.action
        risk_level = generation.risk_level

        self._append_message(session.id, "user", payload.query)
        self._append_message(session.id, "assistant", answer, citations)
        session.active_topic = session_context.active_topic
        session.permission_signature = session_context.access_signature or build_access_signature(user)
        session.summary = summarize_long_history(self.repository.list_messages(session.id))
        session.updated_at = utcnow()
        self.repository.save_session(session)
        if self.session_cache:
            self.session_cache.set_session(session)
            self.session_cache.set_summary(session.id, session.summary)

        latency_ms = int((utcnow() - started_at).total_seconds() * 1000)
        self.audit_service.write_log(
            user=user,
            session_id=session.id,
            request_id=request_id,
            query=payload.query,
            rewritten_query=session_context.rewritten_query,
            scene=payload.scene,
            retrieved=retrieved,
            answer=answer,
            risk_level=risk_level,
            action=risk_action.value,
            latency_ms=latency_ms,
            template=prompt_build.template,
            session_context=session_context,
            input_action=input_risk_action.value,
            output_guard_result=generation.guard_result,
            validation_result=generation.validation_result,
        )

        return ChatQueryResponse(
            request_id=request_id,
            session_id=session.id,
            answer=answer,
            citations=citations,
            risk_action=risk_action,
            retrieved_chunks=len(retrieved),
            rewritten_query=session_context.rewritten_query,
            topic_switched=session_context.topic_switched,
        )

    def list_sessions(self, user: UserContext) -> list[ChatSession]:
        """List sessions that belong to the current user inside the current tenant."""

        sessions = self.repository.list_sessions(user.tenant_id, user.user_id)
        if self.session_cache:
            for session in sessions:
                self.session_cache.set_session(session)
                self.session_cache.set_summary(session.id, session.summary)
        return sessions

    def get_session_detail(self, session_id: str, user: UserContext) -> dict[str, Any]:
        """Return one session and its persisted messages after tenant and user ownership checks."""

        session = self.repository.get_session(session_id)
        if not session and self.session_cache:
            session = self.session_cache.get_session(session_id)
        if not session or session.tenant_id != user.tenant_id or session.user_id != user.user_id:
            raise KeyError(session_id)
        return {"session": session, "messages": self.repository.list_messages(session_id)}

    def _get_or_create_session(self, payload: ChatQueryRequest, user: UserContext) -> ChatSession:
        """Reuse an existing session when allowed, otherwise create a fresh active session."""

        if payload.session_id:
            session = self.repository.get_session(payload.session_id)
            if not session and self.session_cache:
                session = self.session_cache.get_session(payload.session_id)
            if not session or session.tenant_id != user.tenant_id or session.user_id != user.user_id:
                raise KeyError(payload.session_id)
            return session

        now = utcnow()
        session = ChatSession(
            id=f"session_{uuid.uuid4().hex[:12]}",
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            scene=payload.scene,
            status=SessionStatus.ACTIVE,
            permission_signature=build_access_signature(user),
            created_at=now,
            updated_at=now,
        )
        self.repository.save_session(session)
        if self.session_cache:
            self.session_cache.set_session(session)
        return session

    def _append_message(self, session_id: str, role: str, content: str, citations=None) -> None:
        """Persist one chat message into the current session history."""

        self.repository.append_message(
            ChatMessage(
                id=f"msg_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                role=role,
                content=content,
                citations_json=citations or [],
                token_usage=max(len(content.split()), 1),
                created_at=utcnow(),
            )
        )
