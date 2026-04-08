from __future__ import annotations

from app.infrastructure.cache.redis_client import RedisClient


class RateLimitService:
    """Applies simple fixed-window user rate limiting on top of Redis counters."""

    def __init__(self, redis_client: RedisClient, window_seconds: int, max_requests: int) -> None:
        self.redis_client = redis_client
        self.window_seconds = window_seconds
        self.max_requests = max_requests

    def check_user(self, user_id: str, scope: str = "chat") -> tuple[bool, int]:
        """Increment one user counter and return whether the request is still allowed."""

        counter = self.redis_client.increment(
            key=f"rate_limit:user:{user_id}:{scope}",
            ttl_seconds=self.window_seconds,
        )
        return counter <= self.max_requests, counter
