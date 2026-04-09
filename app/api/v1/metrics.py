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
    run = service.run(persist=False)
    payload = run.summary.model_dump()
    trend = service.build_trend_summary(current_run=run)
    payload["trend"] = trend.model_dump()
    payload["alerts"] = [alert.model_dump() for alert in trend.alerts]
    payload["release_readiness"] = service.build_release_readiness_report().model_dump()
    return payload
