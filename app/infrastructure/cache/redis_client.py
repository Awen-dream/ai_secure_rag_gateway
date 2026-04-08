from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

try:
    import redis
except ImportError:  # pragma: no cover - import path is validated in integration/runtime checks.
    redis = None


class RedisClient:
    """Redis adapter with a process-local fallback used for caching and rate limiting."""

    _local_lock = threading.Lock()
    _local_store: dict[str, tuple[str, Optional[float]]] = {}
    _local_queues: dict[str, list[str]] = {}

    def __init__(self, mode: str = "local-fallback", url: str | None = None) -> None:
        self.mode = mode
        self.url = url
        self._client = redis.Redis.from_url(url, decode_responses=True) if self.can_execute() else None

    def can_execute(self) -> bool:
        """Return whether a real Redis backend is configured and the dependency is available."""

        return self.mode == "redis" and bool(self.url) and redis is not None

    def ping(self) -> bool:
        """Return whether the active Redis backend is reachable."""

        if self.can_execute():
            try:
                return bool(self._client.ping())
            except Exception:
                return False
        return True

    def get_json(self, key: str) -> Any:
        """Load one JSON payload from Redis or the local fallback store."""

        raw = self._get_value(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store one JSON payload with TTL semantics."""

        self._set_value(key, json.dumps(value, ensure_ascii=False), ttl_seconds)

    def get_text(self, key: str) -> Optional[str]:
        """Load one plain-text value from Redis or the local fallback store."""

        return self._get_value(key)

    def set_text(self, key: str, value: str, ttl_seconds: int) -> None:
        """Store one plain-text value with TTL semantics."""

        self._set_value(key, value, ttl_seconds)

    def increment(self, key: str, ttl_seconds: int) -> int:
        """Atomically increment one counter and ensure it expires after the provided TTL."""

        if self.can_execute():
            value = int(self._client.incr(key))
            if value == 1:
                self._client.expire(key, ttl_seconds)
            return value

        now = time.time()
        with self._local_lock:
            current_value, expires_at = self._local_store.get(key, ("0", None))
            if expires_at is not None and expires_at <= now:
                current_value = "0"
            new_value = int(current_value) + 1
            self._local_store[key] = (str(new_value), now + ttl_seconds)
            return new_value

    def enqueue_json(self, key: str, value: Any) -> int:
        """Append one JSON payload to a FIFO queue stored in Redis or the local fallback."""

        raw = json.dumps(value, ensure_ascii=False)
        if self.can_execute():
            return int(self._client.rpush(key, raw))

        with self._local_lock:
            queue = self._local_queues.setdefault(key, [])
            queue.append(raw)
            return len(queue)

    def dequeue_json(self, key: str, timeout_seconds: float = 0.0) -> Any:
        """Pop one JSON payload from a FIFO queue, optionally waiting up to the provided timeout."""

        if self.can_execute():
            if timeout_seconds and timeout_seconds > 0:
                result = self._client.blpop(key, timeout=int(timeout_seconds))
                if not result:
                    return None
                _, raw = result
                return json.loads(raw)

            raw = self._client.lpop(key)
            return json.loads(raw) if raw else None

        deadline = time.time() + max(timeout_seconds, 0.0)
        while True:
            with self._local_lock:
                queue = self._local_queues.get(key, [])
                if queue:
                    raw = queue.pop(0)
                    return json.loads(raw)
            if timeout_seconds <= 0:
                return None
            if time.time() >= deadline:
                return None
            time.sleep(min(0.05, timeout_seconds))

    def get_queue_length(self, key: str) -> int:
        """Return the current queue depth for one Redis-backed or local-fallback FIFO queue."""

        if self.can_execute():
            return int(self._client.llen(key))

        with self._local_lock:
            return len(self._local_queues.get(key, []))

    def _get_value(self, key: str) -> Optional[str]:
        if self.can_execute():
            return self._client.get(key)

        now = time.time()
        with self._local_lock:
            item = self._local_store.get(key)
            if not item:
                return None
            value, expires_at = item
            if expires_at is not None and expires_at <= now:
                self._local_store.pop(key, None)
                return None
            return value

    def _set_value(self, key: str, value: str, ttl_seconds: int) -> None:
        if self.can_execute():
            self._client.set(key, value, ex=ttl_seconds)
            return

        with self._local_lock:
            self._local_store[key] = (value, time.time() + ttl_seconds)
