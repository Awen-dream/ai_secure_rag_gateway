from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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


class SourceSyncJob(BaseModel):
    id: str
    tenant_id: str
    provider: str
    name: str
    created_by: str
    source_root: Optional[str] = None
    space_id: Optional[str] = None
    parent_node_token: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = 20
    continue_on_error: bool = True
    default_owner_id: Optional[str] = None
    default_department_scope: list[str] = Field(default_factory=list)
    default_visibility_scope: list[str] = Field(default_factory=lambda: ["tenant"])
    default_security_level: int = 1
    default_tags: list[str] = Field(default_factory=list)
    default_async_mode: bool = True
    enabled: bool = True
    status: str = "idle"
    last_error: Optional[str] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    managed_source_document_ids: list[str] = Field(default_factory=list)
    cycle_seen_source_document_ids: list[str] = Field(default_factory=list)
    last_run_id: Optional[str] = None
    last_run_status: Optional[str] = None
    last_run_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
