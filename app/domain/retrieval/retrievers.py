import re

from app.domain.documents.models import DocumentChunk, DocumentRecord


def normalize_terms(query: str) -> list[str]:
    raw_terms = [term for term in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", query.lower()) if term]
    normalized: list[str] = []
    for term in raw_terms:
        normalized.append(term)
        if re.search(r"[\u4e00-\u9fff]", term) and len(term) > 1:
            normalized.extend(term[index : index + 2] for index in range(len(term) - 1))
    return list(dict.fromkeys(normalized))


def keyword_score(terms: list[str], document: DocumentRecord, chunk: DocumentChunk) -> float:
    haystack = f"{document.title} {chunk.section_name} {chunk.text}".lower()
    score = 0.0
    for term in terms:
        if term in document.title.lower():
            score += 3.0
        if term in haystack:
            score += 1.0
    return score
