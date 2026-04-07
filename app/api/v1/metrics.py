from fastapi import APIRouter, Depends

from app.api.deps import get_audit_service
from app.core.security import require_admin
from app.domain.audit.services import AuditService
from app.domain.auth.models import UserContext

router = APIRouter()


@router.get("/retrieval")
def retrieval_metrics(
    _: UserContext = Depends(require_admin),
    service: AuditService = Depends(get_audit_service),
) -> dict:
    return service.retrieval_metrics().dict()


@router.get("/risk")
def risk_metrics(
    _: UserContext = Depends(require_admin),
    service: AuditService = Depends(get_audit_service),
) -> dict:
    return service.risk_metrics()
