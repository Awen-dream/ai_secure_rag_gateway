from app.domain.retrieval.models import RetrievalProfile


def get_retrieval_profile(intent: str) -> RetrievalProfile:
    profiles = {
        "exact_lookup": RetrievalProfile(
            name="exact_lookup",
            keyword_weight=0.75,
            vector_weight=0.25,
            title_boost=0.15,
            min_score=0.3,
            relative_score_cutoff=0.55,
            top_k=5,
            candidate_pool=10,
        ),
        "summary": RetrievalProfile(
            name="summary",
            keyword_weight=0.35,
            vector_weight=0.65,
            title_boost=0.05,
            min_score=0.16,
            relative_score_cutoff=0.3,
            top_k=6,
            candidate_pool=14,
        ),
        "standard_qa": RetrievalProfile(
            name="standard_qa",
            keyword_weight=0.55,
            vector_weight=0.45,
            title_boost=0.08,
            min_score=0.22,
            relative_score_cutoff=0.5,
            top_k=5,
            candidate_pool=12,
        ),
    }
    return profiles.get(intent, profiles["standard_qa"])
