from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.infrastructure.cache.redis_client import RedisClient

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.utcnow()


class DocumentIngestionTaskQueue:
    """Queues document ingestion work for execution by a dedicated worker process."""

    def __init__(self, redis_client: RedisClient, queue_name: str) -> None:
        self.redis_client = redis_client
        self.queue_name = queue_name

    def enqueue_document(self, doc_id: str) -> dict[str, str]:
        """Enqueue one document ingestion job and return a task receipt for audit or polling."""

        task = {
            "task_id": f"task_{uuid.uuid4().hex[:12]}",
            "task_name": "document_ingestion",
            "doc_id": doc_id,
            "queued_at": utcnow().isoformat(),
        }
        self.redis_client.enqueue_json(self.queue_name, task)
        return {
            "task_id": task["task_id"],
            "doc_id": doc_id,
            "status": "queued",
        }

    def dequeue_document(self, timeout_seconds: float = 0.0) -> dict | None:
        """Pop one queued document ingestion task, optionally waiting for new work."""

        payload = self.redis_client.dequeue_json(self.queue_name, timeout_seconds=timeout_seconds)
        if payload is None:
            return None
        return dict(payload)

    def queue_depth(self) -> int:
        """Return the current number of queued ingestion tasks."""

        return self.redis_client.get_queue_length(self.queue_name)

    def health(self) -> dict:
        """Return queue runtime information for admin diagnostics."""

        return {
            "backend": "redis",
            "execute_enabled": self.redis_client.can_execute(),
            "reachable": self.redis_client.ping(),
            "queue_name": self.queue_name,
            "queue_depth": self.queue_depth(),
            "mode": self.redis_client.mode,
        }


class DocumentIngestionWorker:
    """Consumes queued document ingestion tasks and executes the ingestion orchestrator."""

    def __init__(
        self,
        task_queue: DocumentIngestionTaskQueue,
        orchestrator: DocumentIngestionOrchestrator,
        poll_seconds: float = 1.0,
    ) -> None:
        self.task_queue = task_queue
        self.orchestrator = orchestrator
        self.poll_seconds = max(poll_seconds, 0.1)

    def process_once(self, timeout_seconds: float = 0.0) -> dict | None:
        """Consume at most one queued task and return a compact processing summary."""

        task = self.task_queue.dequeue_document(timeout_seconds=timeout_seconds)
        if not task:
            return None

        doc_id = task["doc_id"]
        document = self.orchestrator.process_document(doc_id)
        return {
            "task_id": task["task_id"],
            "doc_id": doc_id,
            "status": document.status.value,
            "last_error": document.last_error,
        }

    def run_forever(self) -> None:
        """Run an endless worker loop that blocks waiting for new queued ingestion tasks."""

        logger.info("Starting document ingestion worker for queue %s", self.task_queue.queue_name)
        while True:
            result = self.process_once(timeout_seconds=self.poll_seconds)
            if result is None:
                continue
            logger.info(
                "Processed document ingestion task %s for %s with status=%s",
                result["task_id"],
                result["doc_id"],
                result["status"],
            )
            if result["status"] == "failed":
                logger.warning(
                    "Document ingestion task %s failed for %s: %s",
                    result["task_id"],
                    result["doc_id"],
                    result["last_error"],
                )


def enqueue(task_name: str, payload: dict) -> dict:
    """Compatibility helper retained while the rest of the app moves to task-queue services."""

    return {
        "task": task_name,
        "payload": payload,
        "status": "queued",
        "queued_at": utcnow().isoformat(),
    }
