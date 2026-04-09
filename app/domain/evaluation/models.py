from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    id: str
    query: str
    scene: str = "standard_qa"
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_titles: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_intent: Optional[str] = None
    tenant_id: str = "eval"
    user_id: str = "eval_user"
    department_id: str = "engineering"
    role: str = "admin"
    clearance_level: int = 3
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    sample_id: str
    query: str
    scene: str
    hit_expected_doc: bool = False
    hit_expected_title: bool = False
    answer_contains_expected: bool = False
    answer_valid: bool = False
    matched_doc_ids: list[str] = Field(default_factory=list)
    matched_titles: list[str] = Field(default_factory=list)
    retrieved_chunks: int = 0
    rewritten_query: str = ""
    intent: str = ""
    latency_ms: int = 0
    validation_missing_sections: list[str] = Field(default_factory=list)
    answer_preview: str = ""
    citations: list[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    total_cases: int = 0
    retrieval_hit_rate: float = 0.0
    title_hit_rate: float = 0.0
    answer_match_rate: float = 0.0
    answer_valid_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_retrieved_chunks: float = 0.0


class EvalRunResult(BaseModel):
    dataset_size: int
    started_at: datetime
    finished_at: datetime
    summary: EvalRunSummary
    cases: list[EvalCaseResult] = Field(default_factory=list)
