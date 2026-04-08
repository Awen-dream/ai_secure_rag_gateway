from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    title: str
    content: str
    source_type: str = "manual"
    source_uri: Optional[str] = None
    source_connector: Optional[str] = None
    source_document_id: Optional[str] = None
    source_document_version: Optional[str] = None
    owner_id: Optional[str] = None
    department_scope: List[str] = Field(default_factory=list)
    visibility_scope: List[str] = Field(default_factory=lambda: ["tenant"])
    security_level: int = Field(default=1, ge=0, le=10)
    tags: List[str] = Field(default_factory=list)
    async_mode: bool = False
