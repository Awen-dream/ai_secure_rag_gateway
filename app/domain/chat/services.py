from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.application.conversation.summarizer import summarize_recent_messages
from app.domain.audit.services import AuditService
from app.domain.auth.models import UserContext
from app.domain.chat.models import ChatMessage, ChatSession, SessionStatus
from app.domain.chat.schemas import ChatQueryRequest, ChatQueryResponse
from app.domain.citations.services import build_citations
from app.domain.prompts.services import PromptService
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import RiskAction
from app.domain.risk.services import PolicyEngine
from app.infrastructure.db.repositories.memory import store


def utcnow() -> datetime:
    return datetime.utcnow()


class ChatService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_service: PromptService,
        policy_engine: PolicyEngine,
        audit_service: AuditService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.prompt_service = prompt_service
        self.policy_engine = policy_engine
        self.audit_service = audit_service

    def query(self, payload: ChatQueryRequest, user: UserContext) -> ChatQueryResponse:
        started_at = utcnow()
        session = self._get_or_create_session(payload, user)
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        template = self.prompt_service.get_template(payload.scene)
        retrieved = self.retrieval_service.retrieve(user, payload.query)
        risk_action, risk_level = self.policy_engine.evaluate(user, payload.query, len(retrieved))
        citations = build_citations(retrieved)
        answer = self._build_answer(payload.query, template.name, retrieved, citations, risk_action)

        self._append_message(session.id, "user", payload.query)
        self._append_message(session.id, "assistant", answer, citations)
        session.summary = summarize_recent_messages(store.chat_messages.get(session.id, []))
        session.updated_at = utcnow()

        latency_ms = int((utcnow() - started_at).total_seconds() * 1000)
        self.audit_service.write_log(
            user=user,
            session_id=session.id,
            request_id=request_id,
            query=payload.query,
            retrieved=retrieved,
            answer=answer,
            risk_level=risk_level,
            action=risk_action.value,
            latency_ms=latency_ms,
        )

        return ChatQueryResponse(
            request_id=request_id,
            session_id=session.id,
            answer=answer,
            citations=citations,
            risk_action=risk_action,
            retrieved_chunks=len(retrieved),
        )

    def list_sessions(self, user: UserContext) -> list[ChatSession]:
        return [
            session
            for session in store.chat_sessions.values()
            if session.tenant_id == user.tenant_id and session.user_id == user.user_id
        ]

    def get_session_detail(self, session_id: str, user: UserContext) -> dict[str, Any]:
        session = store.chat_sessions.get(session_id)
        if not session or session.tenant_id != user.tenant_id or session.user_id != user.user_id:
            raise KeyError(session_id)
        return {"session": session, "messages": store.chat_messages.get(session_id, [])}

    def _get_or_create_session(self, payload: ChatQueryRequest, user: UserContext) -> ChatSession:
        if payload.session_id:
            session = store.chat_sessions.get(payload.session_id)
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
            created_at=now,
            updated_at=now,
        )
        store.chat_sessions[session.id] = session
        store.chat_messages[session.id] = []
        return session

    @staticmethod
    def _append_message(session_id: str, role: str, content: str, citations=None) -> None:
        store.chat_messages.setdefault(session_id, []).append(
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

    @staticmethod
    def _build_answer(query: str, template_name: str, retrieved, citations, risk_action: RiskAction) -> str:
        if risk_action == RiskAction.REFUSE:
            return (
                "结论：当前请求触发了安全策略，平台已拒绝回答。\n"
                "依据：问题中包含高风险指令或疑似 Prompt Injection 特征。"
            )
        if not retrieved:
            if risk_action == RiskAction.CITATIONS_ONLY:
                return "结论：根据当前已授权资料无法确认。\n依据：高敏部门在无证据命中时只返回保守结果。"
            return "结论：根据当前已授权资料无法确认。\n依据：检索范围内没有找到足够证据。"

        evidence_lines = [f"[{item.index}] {result.chunk.text.replace(chr(10), ' ')[:180]}" for item, result in zip(citations, retrieved)]
        citation_text = ", ".join(f"[{item.index}] {item.title}" for item in citations)
        return (
            "结论：已基于授权知识范围给出回答。\n"
            f"依据：问题“{query}”命中了 {len(retrieved)} 个权限内知识片段，模板策略为 {template_name}。\n"
            + "\n".join(evidence_lines)
            + f"\n引用来源：{citation_text}"
        )
