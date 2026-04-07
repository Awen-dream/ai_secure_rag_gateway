from app.application.query.intent import classify_query_intent
from app.application.query.rewrite import rewrite_query
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.models import RetrievalProfile
from app.domain.retrieval.models import RetrievalResult
from app.domain.retrieval.profiles import get_retrieval_profile
from app.domain.retrieval.rerankers import sort_by_score
from app.domain.retrieval.rerankers import weighted_fusion
from app.domain.retrieval.retrievers import keyword_features, normalize_terms, vector_score


class RetrievalService:
    def __init__(self, document_service: DocumentService) -> None:
        self.document_service = document_service

    def retrieve(self, user: UserContext, query: str, top_k: int = 5) -> list[RetrievalResult]:
        rewritten = rewrite_query(query)
        intent = classify_query_intent(rewritten)
        profile = get_retrieval_profile(intent)
        terms = normalize_terms(rewritten)
        results = self._hybrid_retrieve(user, rewritten, terms, profile)
        return sort_by_score(results)[: min(top_k, profile.top_k)]

    def _hybrid_retrieve(
        self,
        user: UserContext,
        query: str,
        terms: list[str],
        profile: RetrievalProfile,
    ) -> list[RetrievalResult]:
        raw_candidates: list[RetrievalResult] = []
        max_keyword = 0.0
        max_vector = 0.0

        for document, chunk in self.document_service.get_accessible_chunks(user):
            keyword_raw, matched_terms = keyword_features(terms, document, chunk)
            vector_raw = vector_score(query, document, chunk)
            if keyword_raw <= 0 and vector_raw <= 0:
                continue
            max_keyword = max(max_keyword, keyword_raw)
            max_vector = max(max_vector, vector_raw)
            raw_candidates.append(
                RetrievalResult(
                    document=document,
                    chunk=chunk,
                    score=0.0,
                    keyword_score=keyword_raw,
                    vector_score=vector_raw,
                    matched_terms=matched_terms,
                    retrieval_sources=self._resolve_sources(keyword_raw, vector_raw),
                )
            )

        normalized_candidates: list[RetrievalResult] = []
        for item in raw_candidates:
            keyword_normalized = item.keyword_score / max_keyword if max_keyword else 0.0
            vector_normalized = item.vector_score / max_vector if max_vector else 0.0
            title_boost = profile.title_boost if any(term in item.document.title.lower() for term in terms) else 0.0
            item.score = weighted_fusion(
                keyword_score=keyword_normalized,
                vector_score=vector_normalized,
                keyword_weight=profile.keyword_weight,
                vector_weight=profile.vector_weight,
                title_boost=title_boost,
            )
            item.keyword_score = round(keyword_normalized, 4)
            item.vector_score = round(vector_normalized, 4)
            normalized_candidates.append(item)

        ranked_candidates = sort_by_score(normalized_candidates)
        if not ranked_candidates:
            return []

        top_score = ranked_candidates[0].score
        filtered_candidates = [
            item
            for item in ranked_candidates
            if item.score >= profile.min_score and item.score >= top_score * profile.relative_score_cutoff
        ]
        return filtered_candidates[: profile.candidate_pool]

    @staticmethod
    def _resolve_sources(keyword_raw: float, vector_raw: float) -> list[str]:
        sources: list[str] = []
        if keyword_raw > 0:
            sources.append("keyword")
        if vector_raw > 0:
            sources.append("vector")
        return sources
