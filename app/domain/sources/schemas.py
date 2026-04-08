from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FeishuImportRequest(BaseModel):
    source: str
    title: Optional[str] = None
    owner_id: Optional[str] = None
    department_scope: List[str] = Field(default_factory=list)
    visibility_scope: List[str] = Field(default_factory=lambda: ["tenant"])
    security_level: int = Field(default=1, ge=0, le=10)
    tags: List[str] = Field(default_factory=list)
    async_mode: bool = True


class FeishuImportResponse(BaseModel):
    source: str
    source_kind: str
    document_id: str
    document_status: str
    queued: bool
    task_id: Optional[str] = None
