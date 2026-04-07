def enqueue_document_ingestion(doc_id: str) -> dict:
    return {"doc_id": doc_id, "status": "queued"}
