from fastapi import APIRouter, Depends

from app.api.deps import get_audit_service, get_policy_engine, get_prompt_service
from app.core.security import require_admin
from app.domain.auth.models import UserContext
from app.domain.audit.services import AuditService
from app.domain.prompts.models import PromptTemplate
from app.domain.prompts.services import PromptService
from app.domain.risk.models import PolicyDefinition
from app.domain.risk.services import PolicyEngine

router = APIRouter()


@router.get("/prompts", response_model=list[PromptTemplate])
def list_prompt_templates(
    _: UserContext = Depends(require_admin),
    service: PromptService = Depends(get_prompt_service),
) -> list[PromptTemplate]:
    return service.list_templates()


@router.post("/prompts", response_model=PromptTemplate)
def create_prompt_template(
    payload: PromptTemplate,
    _: UserContext = Depends(require_admin),
    service: PromptService = Depends(get_prompt_service),
) -> PromptTemplate:
    return service.add_template(payload)


@router.get("/policies", response_model=list[PolicyDefinition])
def list_policies(
    _: UserContext = Depends(require_admin),
    service: PolicyEngine = Depends(get_policy_engine),
) -> list[PolicyDefinition]:
    return service.list_policies()


@router.post("/policies", response_model=PolicyDefinition)
def create_policy(
    payload: PolicyDefinition,
    _: UserContext = Depends(require_admin),
    service: PolicyEngine = Depends(get_policy_engine),
) -> PolicyDefinition:
    return service.add_policy(payload)


@router.get("/audit")
def list_audit_logs(
    _: UserContext = Depends(require_admin),
    service: AuditService = Depends(get_audit_service),
) -> list[dict]:
    return [log.dict() for log in service.list_logs()]
