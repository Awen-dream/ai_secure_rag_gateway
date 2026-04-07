from app.domain.retrieval.models import RetrievalResult


def sort_by_score(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(results, key=lambda item: item.score, reverse=True)


def weighted_fusion(
    keyword_score: float,
    vector_score: float,
    keyword_weight: float,
    vector_weight: float,
    title_boost: float = 0.0,
) -> float:
    return (keyword_score * keyword_weight) + (vector_score * vector_weight) + title_boost
