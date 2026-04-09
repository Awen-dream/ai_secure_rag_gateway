from __future__ import annotations

from datetime import datetime

from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.domain.auth.models import UserContext
from app.domain.evaluation.models import EvalCaseResult, EvalRunResult, EvalRunSummary, EvalSample
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import RiskAction
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore


def utcnow() -> datetime:
    return datetime.utcnow()


class OfflineEvaluationService:
    """Run offline evaluation over the current retrieval and generation stack."""

    def __init__(
        self,
        dataset_store: LocalEvalDatasetStore,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
    ) -> None:
        self.dataset_store = dataset_store
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generation_service = generation_service

    def list_samples(self) -> list[EvalSample]:
        return self.dataset_store.list_samples()

    def run(self, limit: int | None = None) -> EvalRunResult:
        started_at = utcnow()
        samples = self.dataset_store.list_samples()
        if limit is not None:
            samples = samples[:limit]

        case_results = [self._run_case(sample) for sample in samples]
        finished_at = utcnow()
        return EvalRunResult(
            dataset_size=len(samples),
            started_at=started_at,
            finished_at=finished_at,
            summary=self._summarize(case_results),
            cases=case_results,
        )

    def _run_case(self, sample: EvalSample) -> EvalCaseResult:
        started_at = utcnow()
        user = UserContext(
            user_id=sample.user_id,
            tenant_id=sample.tenant_id,
            department_id=sample.department_id,
            role=sample.role,
            clearance_level=sample.clearance_level,
        )
        explanation = self.retrieval_service.explain(user, sample.query, top_k=5)
        assembled_context = self.context_builder.build(explanation.results)
        prompt_build = self.prompt_builder.build_chat_prompt(
            scene=sample.scene,
            query=explanation.rewritten_query,
            assembled_context=assembled_context,
            session_summary="",
        )
        generation = self.generation_service.generate_chat_answer(
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
    def _summarize(cases: list[EvalCaseResult]) -> EvalRunSummary:
        total = len(cases)
        if total == 0:
            return EvalRunSummary()
        retrieval_hits = sum(1 for case in cases if case.hit_expected_doc)
        title_hits = sum(1 for case in cases if case.hit_expected_title)
        answer_hits = sum(1 for case in cases if case.answer_contains_expected)
        valid_answers = sum(1 for case in cases if case.answer_valid)
        avg_latency = sum(case.latency_ms for case in cases) / total
        avg_retrieved = sum(case.retrieved_chunks for case in cases) / total
        return EvalRunSummary(
            total_cases=total,
            retrieval_hit_rate=round(retrieval_hits / total, 3),
            title_hit_rate=round(title_hits / total, 3),
            answer_match_rate=round(answer_hits / total, 3),
            answer_valid_rate=round(valid_answers / total, 3),
            average_latency_ms=round(avg_latency, 2),
            average_retrieved_chunks=round(avg_retrieved, 2),
        )
