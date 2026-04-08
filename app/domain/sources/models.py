from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceSyncRun(BaseModel):
    id: str
    tenant_id: str
    provider: str
    triggered_by: str
    mode: str = "manual"
    continue_on_error: bool = True
    request_json: dict[str, Any] = Field(default_factory=dict)
    result_items_json: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    imported_new: int = 0
    reused_current: int = 0
    created_new_version: int = 0
    queued: int = 0
    status: str = "success"
    created_at: datetime
