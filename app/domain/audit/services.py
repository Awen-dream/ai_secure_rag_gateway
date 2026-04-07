import uuid
from datetime import datetime

from app.domain.audit.models import AuditLog, RetrievalMetrics
from app.domain.auth.models import UserContext
from app.domain.retrieval.models import RetrievalResult
from app.domain.risk.models import RiskAction
from app.infrastructure.db.repositories.memory import store


def utcnow() -> datetime:
    return datetime.utcnow()


class AuditService:
    def write_log(
        self,
        user: UserContext,
        session_id: str,
        request_id: str,
        query: str,
        retrieved: list[RetrievalResult],
        answer: str,
        risk_level: str,
        action: str,
        latency_ms: int,
    ) -> None:
        store.audit_logs.append(
            AuditLog(
                id=f"audit_{uuid.uuid4().hex[:12]}",
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                session_id=session_id,
                request_id=request_id,
                query=query,
                retrieval_docs_json=[
                    {
                        "doc_id": result.document.id,
                        "title": result.document.title,
                        "chunk_id": result.chunk.id,
                        "section_name": result.chunk.section_name,
                    }
                    for result in retrieved
                ],
                response_summary=answer[:240],
                action=action,
                risk_level=risk_level,
                latency_ms=latency_ms,
                created_at=utcnow(),
            )
        )

    def list_logs(self) -> list[AuditLog]:
        return store.audit_logs

    def retrieval_metrics(self) -> RetrievalMetrics:
        total = len(store.audit_logs)
        if total == 0:
            return RetrievalMetrics()
        citation_hits = sum(1 for log in store.audit_logs if log.retrieval_docs_json)
        refusals = sum(1 for log in store.audit_logs if log.action == RiskAction.REFUSE.value)
        avg_latency = sum(log.latency_ms for log in store.audit_logs) / total
        return RetrievalMetrics(
            total_queries=total,
            citation_coverage_rate=round(citation_hits / total, 3),
            refusal_rate=round(refusals / total, 3),
            average_latency_ms=round(avg_latency, 2),
        )

    def risk_metrics(self) -> dict:
        total = len(store.audit_logs)
        distribution: dict[str, int] = {}
        for log in store.audit_logs:
            distribution[log.risk_level] = distribution.get(log.risk_level, 0) + 1
        return {"total_requests": total, "risk_distribution": distribution}
