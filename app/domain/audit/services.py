from app.application.session.service import SessionContext
import uuid
from datetime import datetime

from app.domain.audit.models import AuditLog, RetrievalMetrics
from app.domain.auth.models import UserContext
from app.domain.prompts.models import PromptTemplate, PromptValidationResult
from app.domain.retrieval.models import RetrievalResult
from app.domain.risk.models import OutputGuardResult
from app.domain.risk.models import RiskAction
from app.infrastructure.db.repositories.base import MetadataRepository


def utcnow() -> datetime:
    return datetime.utcnow()


class AuditService:
    def __init__(self, repository: MetadataRepository) -> None:
        self.repository = repository

    def write_log(
        self,
        user: UserContext,
        session_id: str,
        request_id: str,
        query: str,
        rewritten_query: str,
        scene: str,
        retrieved: list[RetrievalResult],
        answer: str,
        risk_level: str,
        action: str,
        latency_ms: int,
        template: PromptTemplate,
        session_context: SessionContext,
        input_action: str,
        output_guard_result: OutputGuardResult,
        validation_result: PromptValidationResult,
    ) -> None:
        """Persist one structured audit record spanning retrieval, prompt, conversation and risk decisions."""

        understanding = session_context.query_understanding
        self.repository.append_audit_log(
            AuditLog(
                id=f"audit_{uuid.uuid4().hex[:12]}",
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                session_id=session_id,
                request_id=request_id,
                query=query,
                rewritten_query=rewritten_query,
                scene=scene,
                retrieval_docs_json=[
                    {
                        "doc_id": result.document.id,
                        "title": result.document.title,
                        "version": result.document.version,
                        "chunk_id": result.chunk.id,
                        "section_name": result.chunk.section_name,
                        "page_no": result.chunk.page_no,
                        "score": result.score,
                        "keyword_score": result.keyword_score,
                        "vector_score": result.vector_score,
                        "matched_terms": result.matched_terms,
                        "retrieval_sources": result.retrieval_sources,
                        "rerank_source": result.rerank_source,
                        "rerank_notes": result.rerank_notes,
                        "selection_status": result.selection_status,
                        "selection_reasons": result.selection_reasons,
                        "security_level": result.chunk.security_level,
                    }
                    for result in retrieved
                ],
                prompt_json={
                    "scene": scene,
                    "template_id": template.id,
                    "template_version": template.version,
                    "template_name": template.name,
                    "output_schema": template.output_schema,
                },
                risk_json={
                    "input_action": input_action,
                    "final_action": action,
                    "final_risk_level": risk_level,
                    "output_reasons": output_guard_result.reasons,
                    "validation_missing_sections": validation_result.missing_sections,
                },
                conversation_json={
                    "rewritten_query": rewritten_query,
                    "intent": understanding.intent,
                    "intent_confidence": understanding.confidence,
                    "intent_reasons": understanding.reasons,
                    "understanding_source": understanding.source,
                    "rule_rewritten_query": understanding.rule_rewritten_query,
                    "rule_intent": understanding.rule_intent,
                    "rule_intent_confidence": understanding.rule_confidence,
                    "rule_intent_reasons": understanding.rule_reasons,
                    "topic_switched": session_context.topic_switched,
                    "used_history": session_context.used_history,
                    "active_topic": session_context.active_topic,
                    "permission_signature": session_context.access_signature,
                    "session_summary": session_context.session_summary[:240],
                },
                response_summary=answer[:240],
                action=action,
                risk_level=risk_level,
                latency_ms=latency_ms,
                created_at=utcnow(),
            )
        )

    def list_logs(self) -> list[AuditLog]:
        return self.repository.list_audit_logs()

    def retrieval_metrics(self) -> RetrievalMetrics:
        logs = self.repository.list_audit_logs()
        total = len(logs)
        if total == 0:
            return RetrievalMetrics()
        citation_hits = sum(1 for log in logs if log.retrieval_docs_json)
        refusals = sum(1 for log in logs if log.action == RiskAction.REFUSE.value)
        avg_latency = sum(log.latency_ms for log in logs) / total
        average_retrieved_chunks = sum(len(log.retrieval_docs_json) for log in logs) / total
        rewrites = sum(1 for log in logs if log.rewritten_query and log.rewritten_query != log.query)
        return RetrievalMetrics(
            total_queries=total,
            citation_coverage_rate=round(citation_hits / total, 3),
            refusal_rate=round(refusals / total, 3),
            average_latency_ms=round(avg_latency, 2),
            average_retrieved_chunks=round(average_retrieved_chunks, 2),
            rewrite_rate=round(rewrites / total, 3),
        )

    def risk_metrics(self) -> dict:
        logs = self.repository.list_audit_logs()
        total = len(logs)
        distribution: dict[str, int] = {}
        action_distribution: dict[str, int] = {}
        for log in logs:
            distribution[log.risk_level] = distribution.get(log.risk_level, 0) + 1
            action_distribution[log.action] = action_distribution.get(log.action, 0) + 1
        return {
            "total_requests": total,
            "risk_distribution": distribution,
            "action_distribution": action_distribution,
        }
