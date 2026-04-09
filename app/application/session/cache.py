from __future__ import annotations

from app.domain.chat.models import ChatSession
from app.infrastructure.cache.redis_client import RedisClient


class SessionCache:
    """Redis-backed read-through cache for chat session snapshots and summaries."""

    def __init__(self, redis_client: RedisClient, ttl_seconds: int = 3600) -> None:
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"chat:session:{session_id}"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"chat:summary:{session_id}"

    def get_session(self, session_id: str) -> ChatSession | None:
        """Return one cached session snapshot when available."""

        payload = self.redis_client.get_json(self._session_key(session_id))
        return ChatSession.model_validate(payload) if payload else None

    def set_session(self, session: ChatSession) -> None:
        """Cache one session snapshot after mutations or fresh loads."""

        self.redis_client.set_json(self._session_key(session.id), session.model_dump(mode="json"), self.ttl_seconds)

    def get_summary(self, session_id: str) -> str | None:
        """Return one cached summary string when available."""

        return self.redis_client.get_text(self._summary_key(session_id))

    def set_summary(self, session_id: str, summary: str) -> None:
        """Cache one session summary string."""

        self.redis_client.set_text(self._summary_key(session_id), summary, self.ttl_seconds)
