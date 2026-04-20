from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.domain.auth.models import UserContext
from app.domain.evaluation.models import EvalCaseResult, EvalSample
from app.domain.retrieval.rerankers import HeuristicReranker
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import RiskAction


def utcnow() -> datetime:
    return datetime.utcnow()


class EvaluationExecutionEngine(Protocol):
    """Execution engine contract for offline and shadow evaluation runs."""

    engine_name: str

    def run_case(
        self,
        *,
        sample: EvalSample,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
    ) -> EvalCaseResult:
        ...

    def build_shadow_retrieval_service(self, retrieval_service: RetrievalService) -> RetrievalService:
        ...


class NativeEvaluationExecutionEngine:
    """Default evaluation engine that reuses the current retrieval and generation stack."""

    engine_name = "native"

    def run_case(
        self,
        *,
        sample: EvalSample,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
    ) -> EvalCaseResult:
        started_at = utcnow()
        user = UserContext(
            user_id=sample.user_id,
            tenant_id=sample.tenant_id,
            department_id=sample.department_id,
            role=sample.role,
            clearance_level=sample.clearance_level,
        )
        explanation = retrieval_service.explain(user, sample.query, top_k=5)
        assembled_context = context_builder.build(explanation.results)
        prompt_build = prompt_builder.build_chat_prompt(
            scene=sample.scene,
            query=explanation.rewritten_query,
            assembled_context=assembled_context,
            session_summary="",
        )
        generation = generation_service.generate_chat_answer(
            user=user,
            prompt_build=prompt_build,
            input_risk_action=RiskAction.ALLOW,
            input_risk_level="low",
        )

        matched_doc_ids = [
            result.document.id
            for result in explanation.results
            if result.document.id in set(sample.expected_doc_ids)
        ]
        matched_titles = [
            result.document.title
            for result in explanation.results
            if result.document.title in set(sample.expected_titles)
        ]
        answer_contains_expected = all(term in generation.answer for term in sample.expected_answer_contains)
        latency_ms = int((utcnow() - started_at).total_seconds() * 1000)

        return EvalCaseResult(
            sample_id=sample.id,
            query=sample.query,
            scene=sample.scene,
            hit_expected_doc=bool(matched_doc_ids) if sample.expected_doc_ids else False,
            hit_expected_title=bool(matched_titles) if sample.expected_titles else False,
            answer_contains_expected=answer_contains_expected if sample.expected_answer_contains else False,
            answer_valid=generation.validation_result.valid,
            matched_doc_ids=matched_doc_ids,
            matched_titles=matched_titles,
            retrieved_chunks=len(explanation.results),
            rewritten_query=explanation.rewritten_query,
            intent=explanation.intent,
            latency_ms=latency_ms,
            validation_missing_sections=generation.validation_result.missing_sections,
            answer_preview=generation.answer[:240],
            citations=[item.title for item in assembled_context.citations],
        )

    @staticmethod
    def build_shadow_retrieval_service(retrieval_service: RetrievalService) -> RetrievalService:
        required_attributes = (
            "document_service",
            "keyword_backend",
            "vector_backend",
            "query_planning",
            "recall_planning",
        )
        if not all(hasattr(retrieval_service, attribute) for attribute in required_attributes):
            return retrieval_service
        return RetrievalService(
            document_service=retrieval_service.document_service,
            keyword_backend=retrieval_service.keyword_backend,
            vector_backend=retrieval_service.vector_backend,
            retrieval_cache=None,
            reranker=HeuristicReranker(mode="heuristic", top_n=8),
            query_planning=retrieval_service.query_planning,
            recall_planning=retrieval_service.recall_planning,
        )
