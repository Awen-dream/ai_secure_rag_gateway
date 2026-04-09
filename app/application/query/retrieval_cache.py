from __future__ import annotations

import hashlib

from app.domain.auth.models import UserContext
from app.domain.retrieval.models import RetrievalResult
from app.infrastructure.cache.redis_client import RedisClient


class RetrievalCache:
    """Caches permission-scoped retrieval results to reduce repeated backend fan-out."""

    def __init__(self, redis_client: RedisClient, ttl_seconds: int = 300) -> None:
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    def build_key(self, user: UserContext, rewritten_query: str, top_k: int) -> str:
        """Build a permission-aware cache key for one retrieval request."""

        fingerprint = "|".join(
            [
                user.tenant_id,
                user.user_id,
                user.department_id,
                user.role,
                str(user.clearance_level),
                str(top_k),
                rewritten_query.strip(),
            ]
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return f"retrieval:cache:{digest}"

    def get_results(self, user: UserContext, rewritten_query: str, top_k: int) -> list[RetrievalResult] | None:
        """Return cached retrieval results when present."""

        payload = self.redis_client.get_json(self.build_key(user, rewritten_query, top_k))
        if payload is None:
            return None
        return [RetrievalResult.model_validate(item) for item in payload]

    def set_results(
        self,
        user: UserContext,
        rewritten_query: str,
        top_k: int,
        results: list[RetrievalResult],
    ) -> None:
        """Store one retrieval result list for later reuse."""

        self.redis_client.set_json(
            self.build_key(user, rewritten_query, top_k),
            [result.model_dump(mode="json") for result in results],
            self.ttl_seconds,
        )
