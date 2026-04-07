from app.domain.retrieval.models import RetrievalResult


def sort_by_score(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(results, key=lambda item: item.score, reverse=True)
