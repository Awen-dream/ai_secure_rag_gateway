from __future__ import annotations

from dataclasses import dataclass

from app.application.conversation.summarizer import summarize_long_history
from app.application.query.intent import classify_query_intent
from app.application.query.rewrite import rewrite_query
from app.application.query.understanding import QueryUnderstandingResult, QueryUnderstandingService
from app.domain.auth.models import UserContext
from app.domain.chat.models import ChatMessage, ChatSession
from app.infrastructure.db.repositories.base import MetadataRepository


FOLLOW_UP_MARKERS = {
    "它",
    "这个",
    "那个",
    "上面的",
    "刚才",
    "刚刚",
    "上一条",
    "上一轮",
    "前面",
    "继续",
    "审批时限呢",
    "限制呢",
    "流程呢",
}


@dataclass(frozen=True)
class ConversationContext:
    rewritten_query: str
    query_understanding: QueryUnderstandingResult
    session_summary: str
    topic_switched: bool
    active_topic: str
    used_history: bool
    permission_signature: str


def build_permission_signature(user: UserContext) -> str:
    """Render one compact permission signature so context can be reset on auth changes."""

    return "|".join(
        [
            user.tenant_id,
            user.user_id,
            user.department_id,
            user.role,
            str(user.clearance_level),
        ]
    )


class ConversationManager:
    """Owns follow-up rewriting, topic tracking, and permission-aware session context reuse."""

    def __init__(
        self,
        repository: MetadataRepository,
        query_understanding: QueryUnderstandingService | None = None,
    ) -> None:
        self.repository = repository
        self.query_understanding = query_understanding or QueryUnderstandingService()

    def build_context(self, session: ChatSession, user: UserContext, query: str) -> ConversationContext:
        """Rewrite the current query using recent history, topic continuity, and permission boundaries."""

        messages = self.repository.list_messages(session.id)
        permission_signature = build_permission_signature(user)
        permission_changed = bool(session.permission_signature and session.permission_signature != permission_signature)
        summary = "" if permission_changed else (session.summary or summarize_long_history(messages))
        last_user_query = self._last_user_query(messages)
        active_topic = self._infer_topic(query)
        previous_topic = session.active_topic or self._infer_topic(last_user_query)
        topic_switched = self._did_topic_switch(previous_topic, active_topic)
        should_use_history = bool(messages) and not permission_changed and not topic_switched and self._is_follow_up(query)

        understanding = self.query_understanding.understand(
            query,
            last_user_query=last_user_query if should_use_history else None,
            session_summary=summary if should_use_history else None,
        )
        rewritten_query = understanding.rewritten_query

        if permission_changed:
            active_topic = self._infer_topic(query)
            summary = ""

        return ConversationContext(
            rewritten_query=rewritten_query,
            query_understanding=understanding,
            session_summary=summary,
            topic_switched=topic_switched or permission_changed,
            active_topic=active_topic,
            used_history=should_use_history,
            permission_signature=permission_signature,
        )

    @staticmethod
    def _last_user_query(messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    @staticmethod
    def _is_follow_up(query: str) -> bool:
        normalized = query.strip()
        if len(normalized) <= 12:
            return True
        return any(marker in normalized for marker in FOLLOW_UP_MARKERS)

    @staticmethod
    def _infer_topic(query: str) -> str:
        if not query:
            return ""
        normalized = rewrite_query(query).lower()
        intent = classify_query_intent(normalized)
        if any(token in normalized for token in ["审批时限", "限制呢", "流程呢", "怎么走", "呢？", "呢?"]):
            return "follow_up"
        if "报销" in normalized or "采购" in normalized or "制度" in normalized:
            return "policy"
        if "接口" in normalized or "架构" in normalized or "技术" in normalized:
            return "technical"
        if "流程" in normalized or "入职" in normalized or "调岗" in normalized:
            return "process"
        return intent

    @staticmethod
    def _did_topic_switch(previous_topic: str, current_topic: str) -> bool:
        if not previous_topic or not current_topic:
            return False
        if current_topic == "follow_up":
            return False
        return previous_topic != current_topic
