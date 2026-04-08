from __future__ import annotations


def enqueue_document_ingestion(doc_id: str) -> dict[str, str]:
    """Return a lightweight task receipt for background document processing."""

    return {"doc_id": doc_id, "status": "queued"}
