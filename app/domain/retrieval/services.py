from app.application.query.intent import classify_query_intent
from app.application.query.rewrite import rewrite_query
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.rerankers import sort_by_score
from app.domain.retrieval.retrievers import keyword_score, normalize_terms


class RetrievalService:
    def __init__(self, document_service: DocumentService) -> None:
        self.document_service = document_service

    def retrieve(self, user: UserContext, query: str, top_k: int = 5) -> list[RetrievalResult]:
        rewritten = rewrite_query(query)
        intent = classify_query_intent(rewritten)
        terms = normalize_terms(rewritten)
        results: list[RetrievalResult] = []

        for document, chunk in self.document_service.get_accessible_chunks(user):
            score = keyword_score(terms, document, chunk)
            if intent == "exact_lookup" and any(term in document.title.lower() for term in terms):
                score += 1.5
            if score > 0:
                results.append(RetrievalResult(document=document, chunk=chunk, score=score))

        return sort_by_score(results)[:top_k]
