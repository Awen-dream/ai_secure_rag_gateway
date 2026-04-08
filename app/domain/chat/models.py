from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.citations.services import Citation


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ChatSession(BaseModel):
    id: str
    tenant_id: str
    user_id: str
    scene: str
    status: SessionStatus = SessionStatus.ACTIVE
    summary: str = ""
    active_topic: str = ""
    permission_signature: str = ""
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations_json: list[Citation] = Field(default_factory=list)
    token_usage: int = 0
    created_at: datetime
