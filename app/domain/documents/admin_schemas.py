from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DocumentLifecycleUpdateRequest(BaseModel):
    reason: str = Field(default="updated by admin console", min_length=1)


class DocumentReplaceRequest(BaseModel):
    replaced_by_doc_id: str = Field(min_length=1)
    reason: str = Field(default="superseded by newer document", min_length=1)


class DocumentRestoreRequest(BaseModel):
    reason: str = Field(default="restored by admin console", min_length=1)
    source_last_seen_at: Optional[str] = None


class DocumentStaleQueryResponse(BaseModel):
    threshold_days: int
    documents: list[dict] = Field(default_factory=list)
