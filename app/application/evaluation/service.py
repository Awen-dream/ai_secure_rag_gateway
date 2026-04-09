from __future__ import annotations

import uuid
from datetime import datetime

from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.domain.auth.models import UserContext
from app.domain.evaluation.models import (
    EvalCaseResult,
    EvalRegressionAlert,
    EvalRunListItem,
    EvalRunResult,
    EvalRunSummary,
    EvalTrendSummary,
    EvalSample,
    ShadowEvalCaseDiff,
    ShadowEvalRunResult,
)
from app.domain.retrieval.rerankers import HeuristicReranker
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import RiskAction
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore
from app.infrastructure.storage.local_eval_run_store import LocalEvalRunStore


def utcnow() -> datetime:
    return datetime.utcnow()


class OfflineEvaluationService:
    """Run offline evaluation over the current retrieval and generation stack."""

    def __init__(
        self,
        dataset_store: LocalEvalDatasetStore,
        run_store: LocalEvalRunStore,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
    ) -> None:
        self.dataset_store = dataset_store
        self.run_store = run_store
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generation_service = generation_service

    def list_samples(self) -> list[EvalSample]:
        return self.dataset_store.list_samples()

    def list_runs(self, limit: int = 20) -> list[EvalRunListItem]:
        return self.run_store.list_runs(limit=limit)

    def get_run(self, run_id: str) -> dict | None:
        return self.run_store.load_run(run_id)

    def build_trend_summary(
        self,
        current_run: EvalRunResult | None = None,
        history_limit: int = 10,
    ) -> EvalTrendSummary:
        offline_runs = [item for item in self.run_store.list_runs(limit=history_limit) if item.mode == "offline"]
        baseline_item = offline_runs[0] if offline_runs else None
        if current_run is None and baseline_item is not None:
            current_summary = self._summary_from_dict(baseline_item.summary)
            current_run_id = baseline_item.run_id
            baseline_item = offline_runs[1] if len(offline_runs) > 1 else None
        elif current_run is not None:
            current_summary = current_run.summary
            current_run_id = current_run.run_id
        else:
            current_summary = EvalRunSummary()
            current_run_id = ""

        baseline_summary = self._summary_from_dict(baseline_item.summary) if baseline_item else None
        deltas = self._build_deltas(current_summary, baseline_summary)
        alerts = self._build_regression_alerts(current_summary, baseline_summary, deltas)
        return EvalTrendSummary(
            current_run_id=current_run_id,
            baseline_run_id=baseline_item.run_id if baseline_item else "",
            compared_runs=1 + (1 if baseline_item else 0) if current_run_id else (1 if baseline_item else 0),
            current_summary=current_summary,
            baseline_summary=baseline_summary,
            deltas=deltas,
            alerts=alerts,
        )

    def run(self, limit: int | None = None, persist: bool = True) -> EvalRunResult:
        started_at = utcnow()
        samples = self.dataset_store.list_samples()
        if limit is not None:
            samples = samples[:limit]

        case_results = [self._run_case(sample) for sample in samples]
        finished_at = utcnow()
        run = EvalRunResult(
            run_id=f"eval_{uuid.uuid4().hex[:12]}",
            mode="offline",
            dataset_size=len(samples),
            started_at=started_at,
            finished_at=finished_at,
            summary=self._summarize(case_results),
            cases=case_results,
        )
        if persist:
            self.run_store.save_run(run.run_id, run.model_dump(mode="json"))
        return run

    def run_shadow(self, limit: int | None = None, persist: bool = True) -> ShadowEvalRunResult:
        started_at = utcnow()
        samples = self.dataset_store.list_samples()
        if limit is not None:
            samples = samples[:limit]

        primary_cases = [self._run_case(sample, retrieval_service=self.retrieval_service) for sample in samples]
        shadow_service = self._build_shadow_retrieval_service()
        shadow_cases = [self._run_case(sample, retrieval_service=shadow_service) for sample in samples]
        diffs = [
            ShadowEvalCaseDiff(
                sample_id=sample.id,
                query=sample.query,
                primary_hit=primary.hit_expected_doc or primary.hit_expected_title,
                shadow_hit=shadow.hit_expected_doc or shadow.hit_expected_title,
                primary_answer_match=primary.answer_contains_expected,
                shadow_answer_match=shadow.answer_contains_expected,
                changed=(
                    (primary.hit_expected_doc != shadow.hit_expected_doc)
                    or (primary.hit_expected_title != shadow.hit_expected_title)
                    or (primary.answer_contains_expected != shadow.answer_contains_expected)
                    or (primary.rewritten_query != shadow.rewritten_query)
                ),
                primary_rewritten_query=primary.rewritten_query,
                shadow_rewritten_query=shadow.rewritten_query,
            )
            for sample, primary, shadow in zip(samples, primary_cases, shadow_cases)
        ]
        finished_at = utcnow()
        run = ShadowEvalRunResult(
            run_id=f"shadow_{uuid.uuid4().hex[:12]}",
            mode="shadow",
            dataset_size=len(samples),
            started_at=started_at,
            finished_at=finished_at,
            primary_summary=self._summarize(primary_cases),
            shadow_summary=self._summarize(shadow_cases),
            diffs=diffs,
        )
        if persist:
            self.run_store.save_run(run.run_id, run.model_dump(mode="json"))
        return run

    def _run_case(self, sample: EvalSample, retrieval_service: RetrievalService | None = None) -> EvalCaseResult:
        started_at = utcnow()
        user = UserContext(
            user_id=sample.user_id,
            tenant_id=sample.tenant_id,
            department_id=sample.department_id,
            role=sample.role,
            clearance_level=sample.clearance_level,
        )
        active_retrieval = retrieval_service or self.retrieval_service
        explanation = active_retrieval.explain(user, sample.query, top_k=5)
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

    def _build_shadow_retrieval_service(self) -> RetrievalService:
        required_attributes = (
            "document_service",
            "keyword_backend",
            "vector_backend",
            "query_planning",
            "recall_planning",
        )
        if not all(hasattr(self.retrieval_service, attribute) for attribute in required_attributes):
            return self.retrieval_service
        return RetrievalService(
            document_service=self.retrieval_service.document_service,
            keyword_backend=self.retrieval_service.keyword_backend,
            vector_backend=self.retrieval_service.vector_backend,
            retrieval_cache=None,
            reranker=HeuristicReranker(mode="heuristic", top_n=8),
            query_planning=self.retrieval_service.query_planning,
            recall_planning=self.retrieval_service.recall_planning,
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

    @staticmethod
    def _summary_from_dict(payload: dict | None) -> EvalRunSummary:
        if not payload:
            return EvalRunSummary()
        return EvalRunSummary.model_validate(payload)

    @staticmethod
    def _build_deltas(
        current_summary: EvalRunSummary,
        baseline_summary: EvalRunSummary | None,
    ) -> dict:
        if baseline_summary is None:
            return {}
        return {
            "retrieval_hit_rate": round(current_summary.retrieval_hit_rate - baseline_summary.retrieval_hit_rate, 3),
            "title_hit_rate": round(current_summary.title_hit_rate - baseline_summary.title_hit_rate, 3),
            "answer_match_rate": round(current_summary.answer_match_rate - baseline_summary.answer_match_rate, 3),
            "answer_valid_rate": round(current_summary.answer_valid_rate - baseline_summary.answer_valid_rate, 3),
            "average_latency_ms": round(current_summary.average_latency_ms - baseline_summary.average_latency_ms, 2),
            "average_retrieved_chunks": round(
                current_summary.average_retrieved_chunks - baseline_summary.average_retrieved_chunks,
                2,
            ),
        }

    @staticmethod
    def _build_regression_alerts(
        current_summary: EvalRunSummary,
        baseline_summary: EvalRunSummary | None,
        deltas: dict,
    ) -> list[EvalRegressionAlert]:
        if baseline_summary is None:
            return []

        alerts: list[EvalRegressionAlert] = []
        for metric in (
            "retrieval_hit_rate",
            "title_hit_rate",
            "answer_match_rate",
            "answer_valid_rate",
        ):
            delta = float(deltas.get(metric, 0.0))
            if delta <= -0.05:
                current_value = float(getattr(current_summary, metric))
                baseline_value = float(getattr(baseline_summary, metric))
                alerts.append(
                    EvalRegressionAlert(
                        metric=metric,
                        severity="critical" if delta <= -0.1 else "warning",
                        direction="down",
                        current_value=current_value,
                        baseline_value=baseline_value,
                        delta=delta,
                        message=f"{metric} regressed from {baseline_value:.3f} to {current_value:.3f}.",
                    )
                )

        latency_delta = float(deltas.get("average_latency_ms", 0.0))
        if (
            latency_delta >= 100
            and baseline_summary.average_latency_ms > 0
            and current_summary.average_latency_ms >= baseline_summary.average_latency_ms * 1.2
        ):
            alerts.append(
                EvalRegressionAlert(
                    metric="average_latency_ms",
                    severity="warning",
                    direction="up",
                    current_value=current_summary.average_latency_ms,
                    baseline_value=baseline_summary.average_latency_ms,
                    delta=latency_delta,
                    message=(
                        "average_latency_ms increased from "
                        f"{baseline_summary.average_latency_ms:.2f} to {current_summary.average_latency_ms:.2f}."
                    ),
                )
            )
        return alerts
