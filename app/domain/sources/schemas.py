from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceSyncAction(str):
    IMPORTED_NEW = "imported_new"
    REUSED_CURRENT = "reused_current"
    CREATED_NEW_VERSION = "created_new_version"
    FAILED = "failed"


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
    document_version: int
    sync_action: str
    queued: bool
    task_id: Optional[str] = None


class FeishuBatchSyncRequest(BaseModel):
    items: List[FeishuImportRequest] = Field(default_factory=list)
    source_root: Optional[str] = None
    space_id: Optional[str] = None
    parent_node_token: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=200)
    continue_on_error: bool = True
    default_owner_id: Optional[str] = None
    default_department_scope: List[str] = Field(default_factory=list)
    default_visibility_scope: List[str] = Field(default_factory=lambda: ["tenant"])
    default_security_level: int = Field(default=1, ge=0, le=10)
    default_tags: List[str] = Field(default_factory=list)
    default_async_mode: bool = True


class FeishuBatchSyncItemResponse(BaseModel):
    source: str
    source_kind: Optional[str] = None
    success: bool
    sync_action: str
    document_id: Optional[str] = None
    document_status: Optional[str] = None
    document_version: Optional[int] = None
    queued: bool = False
    task_id: Optional[str] = None
    error: Optional[str] = None


class FeishuListSourcesRequest(BaseModel):
    source_root: Optional[str] = None
    space_id: Optional[str] = None
    parent_node_token: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=200)


class FeishuListSourceItemResponse(BaseModel):
    source: str
    source_kind: str
    external_document_id: str
    title: Optional[str] = None
    space_id: Optional[str] = None
    node_token: Optional[str] = None
    parent_node_token: Optional[str] = None
    obj_type: Optional[str] = None
    has_child: bool = False


class FeishuListSourcesResponse(BaseModel):
    listed_count: int
    next_cursor: Optional[str] = None
    items: List[FeishuListSourceItemResponse] = Field(default_factory=list)


class FeishuBatchSyncResponse(BaseModel):
    total: int
    listed_count: int
    succeeded: int
    failed: int
    imported_new: int
    reused_current: int
    created_new_version: int
    queued: int
    next_cursor: Optional[str] = None
    items: List[FeishuBatchSyncItemResponse] = Field(default_factory=list)
