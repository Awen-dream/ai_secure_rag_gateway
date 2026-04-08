from typing import Dict, List

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    id: str
    scene: str
    version: int
    name: str
    content: str
    output_schema: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    created_by: str = "system"


class RenderedPrompt(BaseModel):
    """Represents the final prompt payload passed into the generation layer."""

    instructions: str
    input_text: str


class PromptPreviewRequest(BaseModel):
    scene: str
    query: str
    session_summary: str = ""
    top_k: int = 4


class PromptPreviewResponse(BaseModel):
    scene: str
    template_id: str
    template_version: int
    instructions: str
    input_text: str
    retrieved_chunks: int


class PromptValidationRequest(BaseModel):
    scene: str
    answer: str


class PromptValidationResult(BaseModel):
    template_id: str
    template_version: int
    valid: bool
    missing_sections: List[str] = Field(default_factory=list)
    normalized_answer: str
