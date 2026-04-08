import unittest
from datetime import datetime
from typing import Optional

from app.application.conversation.memory import ConversationManager, build_permission_signature
from app.application.query.understanding import QueryUnderstandingResult
from app.domain.auth.models import UserContext
from app.domain.chat.models import ChatMessage, ChatSession, SessionStatus


class _Repo:
    def __init__(self, messages):
        self._messages = messages

    def list_messages(self, session_id: str):
        return list(self._messages)


class _Understanding:
    def __init__(self, rewritten_query: str) -> None:
        self.rewritten_query = rewritten_query
        self.calls: list[dict] = []

    def understand(
        self,
        query: str,
        last_user_query: Optional[str] = None,
        session_summary: Optional[str] = None,
    ):
        self.calls.append(
            {
                "query": query,
                "last_user_query": last_user_query,
                "session_summary": session_summary,
            }
        )
        return QueryUnderstandingResult(
            rewritten_query=self.rewritten_query,
            intent="standard_qa",
            confidence=0.9,
            reasons=["test_stub"],
            source="llm",
        )


def build_user(role: str = "employee", clearance: int = 2) -> UserContext:
    return UserContext(
        user_id="u1",
        tenant_id="t1",
        department_id="engineering",
        role=role,
        clearance_level=clearance,
    )


class ConversationMemoryTest(unittest.TestCase):
    def test_follow_up_query_reuses_last_user_query(self) -> None:
        messages = [
            ChatMessage(
                id="m1",
                session_id="s1",
                role="user",
                content="报销制度是什么？",
                created_at=datetime(2026, 1, 1),
            ),
            ChatMessage(
                id="m2",
                session_id="s1",
                role="assistant",
                content="报销制度说明。",
                created_at=datetime(2026, 1, 1),
            ),
        ]
        session = ChatSession(
            id="s1",
            tenant_id="t1",
            user_id="u1",
            scene="standard_qa",
            status=SessionStatus.ACTIVE,
            active_topic="policy",
            permission_signature=build_permission_signature(build_user()),
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        manager = ConversationManager(_Repo(messages))

        context = manager.build_context(session, build_user(), "审批时限呢？")

        self.assertIn("报销制度是什么", context.rewritten_query)
        self.assertFalse(context.topic_switched)
        self.assertTrue(context.used_history)

    def test_topic_switch_resets_follow_up_context(self) -> None:
        messages = [
            ChatMessage(
                id="m1",
                session_id="s1",
                role="user",
                content="报销制度是什么？",
                created_at=datetime(2026, 1, 1),
            )
        ]
        session = ChatSession(
            id="s1",
            tenant_id="t1",
            user_id="u1",
            scene="standard_qa",
            status=SessionStatus.ACTIVE,
            active_topic="policy",
            permission_signature=build_permission_signature(build_user()),
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        manager = ConversationManager(_Repo(messages))

        context = manager.build_context(session, build_user(), "这个接口规范在哪里？")

        self.assertEqual(context.rewritten_query, "这个接口规范在哪里？")
        self.assertTrue(context.topic_switched)
        self.assertFalse(context.used_history)

    def test_permission_change_invalidates_old_context(self) -> None:
        messages = [
            ChatMessage(
                id="m1",
                session_id="s1",
                role="user",
                content="报销制度是什么？",
                created_at=datetime(2026, 1, 1),
            )
        ]
        session = ChatSession(
            id="s1",
            tenant_id="t1",
            user_id="u1",
            scene="standard_qa",
            status=SessionStatus.ACTIVE,
            summary="old summary",
            active_topic="policy",
            permission_signature=build_permission_signature(build_user()),
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        manager = ConversationManager(_Repo(messages))

        context = manager.build_context(session, build_user(clearance=4), "审批时限呢？")

        self.assertEqual(context.session_summary, "")
        self.assertTrue(context.topic_switched)
        self.assertFalse(context.used_history)

    def test_follow_up_query_passes_history_into_query_understanding(self) -> None:
        messages = [
            ChatMessage(
                id="m1",
                session_id="s1",
                role="user",
                content="报销制度是什么？",
                created_at=datetime(2026, 1, 1),
            ),
            ChatMessage(
                id="m2",
                session_id="s1",
                role="assistant",
                content="报销制度说明。",
                created_at=datetime(2026, 1, 1),
            ),
        ]
        session = ChatSession(
            id="s1",
            tenant_id="t1",
            user_id="u1",
            scene="standard_qa",
            status=SessionStatus.ACTIVE,
            active_topic="policy",
            permission_signature=build_permission_signature(build_user()),
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )
        understanding = _Understanding("报销制度的审批时限是什么？")
        manager = ConversationManager(_Repo(messages), query_understanding=understanding)

        context = manager.build_context(session, build_user(), "审批时限呢？")

        self.assertEqual(context.rewritten_query, "报销制度的审批时限是什么？")
        self.assertEqual(len(understanding.calls), 1)
        self.assertEqual(understanding.calls[0]["query"], "审批时限呢？")
        self.assertEqual(understanding.calls[0]["last_user_query"], "报销制度是什么？")


if __name__ == "__main__":
    unittest.main()
