from pydantic import BaseModel, Field


class UserContext(BaseModel):
    user_id: str
    tenant_id: str
    department_id: str
    role: str
    clearance_level: int = Field(ge=0, le=10)
