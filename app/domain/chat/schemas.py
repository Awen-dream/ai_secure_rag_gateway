from typing import Optional

from pydantic import BaseModel, Field

from app.domain.citations.services import Citation
from app.domain.risk.models import RiskAction


class ChatQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    scene: str = "standard_qa"


class ChatQueryResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    risk_action: RiskAction
    retrieved_chunks: int = 0
    rewritten_query: str = ""
    topic_switched: bool = False
