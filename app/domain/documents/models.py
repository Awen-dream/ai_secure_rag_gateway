from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    SUCCESS = "success"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    id: str
    tenant_id: str
    title: str
    source_type: str
    source_uri: Optional[str]
    source_connector: Optional[str] = None
    source_document_id: Optional[str] = None
    source_document_version: Optional[str] = None
    owner_id: str
    department_scope: List[str] = Field(default_factory=list)
    visibility_scope: List[str] = Field(default_factory=list)
    security_level: int
    version: int = 1
    status: DocumentStatus = DocumentStatus.PENDING
    last_error: Optional[str] = None
    content_hash: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)
    current: bool = True


class DocumentChunk(BaseModel):
    id: str
    doc_id: str
    tenant_id: str
    chunk_index: int
    page_no: int = 1
    section_name: str
    text: str
    token_count: int
    security_level: int
    department_scope: List[str] = Field(default_factory=list)
    role_scope: List[str] = Field(default_factory=list)
    metadata_json: Dict[str, str] = Field(default_factory=dict)
