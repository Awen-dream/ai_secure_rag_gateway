from pydantic import BaseModel

from app.domain.documents.models import DocumentChunk, DocumentRecord


class RetrievalResult(BaseModel):
    document: DocumentRecord
    chunk: DocumentChunk
    score: float
