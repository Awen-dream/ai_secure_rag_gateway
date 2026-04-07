from fastapi import APIRouter, Depends

from app.api.deps import get_audit_service, get_policy_engine, get_prompt_service, get_retrieval_service
from app.core.security import require_admin
from app.domain.auth.models import UserContext
from app.domain.audit.services import AuditService
from app.domain.prompts.models import PromptTemplate
from app.domain.prompts.services import PromptService
from app.domain.retrieval.models import RetrievalBackendInfo, RetrievalExplainRequest, RetrievalExplainResponse
from app.domain.retrieval.services import RetrievalService
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


@router.get("/retrieval/backends", response_model=list[RetrievalBackendInfo])
def list_retrieval_backends(
    _: UserContext = Depends(require_admin),
    service: RetrievalService = Depends(get_retrieval_service),
) -> list[RetrievalBackendInfo]:
    """Return the active retrieval backends and their runtime configuration summary."""

    return service.backend_info()


@router.post("/retrieval/explain", response_model=RetrievalExplainResponse)
def explain_retrieval(
    payload: RetrievalExplainRequest,
    user: UserContext = Depends(require_admin),
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrievalExplainResponse:
    """Explain hybrid retrieval behavior for an admin-visible query under current access scope."""

    return service.explain(user, payload.query, payload.top_k)
