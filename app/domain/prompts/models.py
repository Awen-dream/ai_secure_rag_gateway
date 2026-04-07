from typing import Dict

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
