from __future__ import annotations

from app.infrastructure.queue.worker import DocumentIngestionTaskQueue


def enqueue_document_ingestion(task_queue: DocumentIngestionTaskQueue, doc_id: str) -> dict[str, str]:
    """Queue one document ingestion task for execution by an external worker."""

    return task_queue.enqueue_document(doc_id)
