from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AuditLog(BaseModel):
    id: str
    user_id: str
    tenant_id: str
    session_id: str
    request_id: str
    query: str
    rewritten_query: str = ""
    scene: str = ""
    retrieval_docs_json: List[Dict[str, Any]] = Field(default_factory=list)
    prompt_json: Dict[str, Any] = Field(default_factory=dict)
    risk_json: Dict[str, Any] = Field(default_factory=dict)
    conversation_json: Dict[str, Any] = Field(default_factory=dict)
    response_summary: str
    action: str
    risk_level: str
    latency_ms: int
    created_at: datetime


class RetrievalMetrics(BaseModel):
    total_queries: int = 0
    citation_coverage_rate: float = 0.0
    refusal_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_retrieved_chunks: float = 0.0
    rewrite_rate: float = 0.0
