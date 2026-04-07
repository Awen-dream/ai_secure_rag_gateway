from app.application.ingestion.tasks import enqueue_document_ingestion


def start_ingestion(doc_id: str) -> dict:
    return enqueue_document_ingestion(doc_id)
