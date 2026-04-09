from fastapi import APIRouter, Depends

from app.api.deps import get_audit_service, get_offline_evaluation_service
from app.core.security import require_admin
from app.domain.audit.services import AuditService
from app.domain.auth.models import UserContext
from app.application.evaluation.service import OfflineEvaluationService

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


@router.get("/evaluation")
def evaluation_metrics(
    _: UserContext = Depends(require_admin),
    service: OfflineEvaluationService = Depends(get_offline_evaluation_service),
) -> dict:
    return service.run().summary.model_dump()
