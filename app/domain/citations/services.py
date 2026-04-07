from typing import Optional

from pydantic import BaseModel

from app.domain.retrieval.models import RetrievalResult


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    section_name: str
    version: int
    source_uri: Optional[str] = None


def build_citations(results: list[RetrievalResult]) -> list[Citation]:
    citations: list[Citation] = []
    seen_docs: set[str] = set()
    for index, result in enumerate(results, start=1):
        if result.document.id in seen_docs:
            continue
        citations.append(
            Citation(
                index=index,
                doc_id=result.document.id,
                title=result.document.title,
                section_name=result.chunk.section_name,
                version=result.document.version,
                source_uri=result.document.source_uri,
            )
        )
        seen_docs.add(result.document.id)
    return citations
