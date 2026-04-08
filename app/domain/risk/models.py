from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RiskAction(str, Enum):
    ALLOW = "allow"
    MASK = "mask"
    REFUSE = "refuse"
    CITATIONS_ONLY = "citations_only"


class PolicyDefinition(BaseModel):
    id: str
    name: str
    description: str
    high_risk_terms: List[str] = Field(default_factory=list)
    restricted_departments: List[str] = Field(default_factory=list)
    enabled: bool = True


class OutputGuardResult(BaseModel):
    """Normalized output-guard decision returned after post-generation safety checks."""

    action: RiskAction = RiskAction.ALLOW
    answer: str
    risk_level: str = "low"
    reasons: List[str] = Field(default_factory=list)
