import math
import re
from collections import Counter

from app.domain.documents.models import DocumentChunk, DocumentRecord


def normalize_terms(query: str) -> list[str]:
    raw_terms = [term for term in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", query.lower()) if term]
    normalized: list[str] = []
    for term in raw_terms:
        normalized.append(term)
        if re.search(r"[\u4e00-\u9fff]", term) and len(term) > 1:
            normalized.extend(term[index : index + 2] for index in range(len(term) - 1))
    return list(dict.fromkeys(normalized))


def keyword_features(terms: list[str], document: DocumentRecord, chunk: DocumentChunk) -> tuple[float, list[str]]:
    haystack = f"{document.title} {chunk.section_name} {chunk.text}".lower()
    title = document.title.lower()
    score = 0.0
    matched_terms: list[str] = []
    for term in terms:
        matched = False
        if term in title:
            score += 3.0
            matched = True
        if term in haystack:
            score += 1.0
            matched = True
        if matched:
            matched_terms.append(term)
    return score, list(dict.fromkeys(matched_terms))


def semantic_features(text: str) -> Counter:
    normalized = text.lower()
    tokens = normalize_terms(normalized)
    features: list[str] = list(tokens)
    compact = "".join(char for char in normalized if re.search(r"[a-z0-9\u4e00-\u9fff]", char))
    if len(compact) > 1:
        features.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return Counter(features)


def vector_score(query: str, document: DocumentRecord, chunk: DocumentChunk) -> float:
    query_features = semantic_features(query)
    target_features = semantic_features(f"{document.title} {chunk.section_name} {chunk.text}")
    if not query_features or not target_features:
        return 0.0

    numerator = sum(query_features[token] * target_features[token] for token in query_features)
    query_norm = math.sqrt(sum(value * value for value in query_features.values()))
    target_norm = math.sqrt(sum(value * value for value in target_features.values()))
    if query_norm == 0 or target_norm == 0:
        return 0.0
    return numerator / (query_norm * target_norm)
