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
    title = document.title.lower()
    section = chunk.section_name.lower()
    body = chunk.text.lower()
    title_terms = normalize_terms(title)
    section_terms = normalize_terms(section)
    body_terms = normalize_terms(body)
    avg_title_len = max(len(title_terms), 6)
    avg_section_len = max(len(section_terms), 8)
    avg_body_len = max(len(body_terms), 80)
    score = 0.0
    matched_terms: list[str] = []
    unique_terms = list(dict.fromkeys(term for term in terms if term))
    for term in unique_terms:
        matched = False
        specificity = _term_specificity(term)
        title_tf = title.count(term)
        section_tf = section.count(term)
        body_tf = body.count(term)

        if title_tf:
            score += 3.2 * _bm25_component(title_tf, len(title_terms), avg_title_len) * specificity
            matched = True
        if section_tf:
            score += 2.1 * _bm25_component(section_tf, len(section_terms), avg_section_len) * specificity
            matched = True
        if body_tf:
            score += 1.0 * _bm25_component(body_tf, len(body_terms), avg_body_len) * specificity
            matched = True

        if len(term) >= 4 and term in body:
            score += 0.35 * specificity
            matched = True
        if len(term) >= 4 and term in title:
            score += 0.55 * specificity
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


def _bm25_component(tf: int, field_len: int, avg_len: int, k1: float = 1.2, b: float = 0.75) -> float:
    if tf <= 0:
        return 0.0
    normalized_len = max(field_len, 1) / max(avg_len, 1)
    denominator = tf + (k1 * (1 - b + (b * normalized_len)))
    if denominator == 0:
        return 0.0
    return ((k1 + 1) * tf) / denominator


def _term_specificity(term: str) -> float:
    if not term:
        return 1.0
    alnum = bool(re.search(r"[a-z0-9]", term))
    cjk = bool(re.search(r"[\u4e00-\u9fff]", term))
    base = 1.0 + min(len(term), 10) * 0.08
    if alnum and any(char.isdigit() for char in term):
        base += 0.35
    if cjk and len(term) >= 4:
        base += 0.2
    return base
