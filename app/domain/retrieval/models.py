from typing import List

from pydantic import BaseModel, Field

from app.domain.documents.models import DocumentChunk, DocumentRecord


class RetrievalProfile(BaseModel):
    name: str
    keyword_weight: float
    vector_weight: float
    title_boost: float = 0.0
    min_score: float = 0.18
    relative_score_cutoff: float = 0.45
    top_k: int = 5
    candidate_pool: int = 12


class RetrievalResult(BaseModel):
    document: DocumentRecord
    chunk: DocumentChunk
    score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0
    matched_terms: List[str] = Field(default_factory=list)
    retrieval_sources: List[str] = Field(default_factory=list)


class RetrievalBackendInfo(BaseModel):
    """Describes one retrieval backend and the deployment-relevant metadata around it."""

    backend: str
    mode: str
    config: dict = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)


class RetrievalExplainRequest(BaseModel):
    """Admin-facing request payload for inspecting hybrid retrieval behavior."""

    query: str
    top_k: int = 5


class RetrievalExplainResponse(BaseModel):
    """Admin-facing hybrid retrieval explanation payload."""

    rewritten_query: str
    intent: str
    profile: RetrievalProfile
    results: List[RetrievalResult] = Field(default_factory=list)


class RetrievalBackendPlan(BaseModel):
    """Admin-facing backend plan payload used for integration preview and debugging."""

    backend: str
    execute_enabled: bool
    artifacts: dict = Field(default_factory=dict)
