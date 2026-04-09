from __future__ import annotations

import uuid
from datetime import datetime

from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.domain.auth.models import UserContext
from app.domain.evaluation.models import (
    EvalCaseResult,
    EvalQualityGate,
    EvalRegressionAlert,
    EvalRunListItem,
    EvalRunResult,
    EvalRunSummary,
    EvalTrendSummary,
    EvalSample,
    ReleaseReadinessReport,
    ShadowReportSummary,
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

    def build_shadow_report(self) -> ShadowReportSummary:
        latest_shadow_payload = self._load_latest_run_payload(mode="shadow")
        if latest_shadow_payload is None:
            return ShadowReportSummary()

        winner = str(latest_shadow_payload.get("winner", "unavailable"))
        changed_cases = sum(1 for item in latest_shadow_payload.get("diffs", []) if item.get("changed"))
        recommendation = "review"
        if winner == "primary":
            recommendation = "keep_primary"
        elif winner == "shadow":
            recommendation = "investigate_shadow"
        elif winner == "tie":
            recommendation = "keep_primary"

        return ShadowReportSummary(
            latest_run_id=str(latest_shadow_payload.get("run_id", "")),
            winner=winner,
            recommendation=recommendation,
            primary_wins=int(latest_shadow_payload.get("primary_wins", 0)),
            shadow_wins=int(latest_shadow_payload.get("shadow_wins", 0)),
            ties=int(latest_shadow_payload.get("ties", 0)),
            changed_cases=changed_cases,
            winner_reasons=list(latest_shadow_payload.get("winner_reasons", [])),
        )

    def build_release_readiness_report(self) -> ReleaseReadinessReport:
        latest_offline_payload = self._load_latest_run_payload(mode="offline")
        if latest_offline_payload is None:
            return ReleaseReadinessReport(
                generated_at=utcnow(),
                decision="hold",
                reasons=["No offline evaluation run is available."],
            )

        latest_offline_run = EvalRunResult.model_validate(latest_offline_payload)
        trend = self.build_trend_summary(current_run=latest_offline_run)
        shadow_report = self.build_shadow_report()

        decision = "ready"
        reasons: list[str] = []
        if latest_offline_run.quality_gate.status == "block" or trend.quality_gate.status == "block":
            decision = "hold"
            reasons.append("Offline evaluation quality gate is blocking release.")
        elif latest_offline_run.quality_gate.status == "warning" or trend.quality_gate.status == "warning":
            decision = "review"
            reasons.append("Offline evaluation quality gate requires review.")

        if shadow_report.winner == "shadow":
            if decision == "ready":
                decision = "review"
            reasons.append("Shadow baseline currently outperforms the primary stack.")
        elif shadow_report.winner == "unavailable":
            if decision == "ready":
                decision = "review"
            reasons.append("No shadow evaluation run is available.")

        if not reasons:
            reasons.append("Offline evaluation and shadow comparison both meet release expectations.")

        return ReleaseReadinessReport(
            generated_at=utcnow(),
            decision=decision,
            latest_offline_run_id=latest_offline_run.run_id,
            latest_shadow_run_id=shadow_report.latest_run_id,
            quality_gate=latest_offline_run.quality_gate,
            trend=trend,
            shadow_report=shadow_report,
            reasons=reasons,
        )

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
        quality_gate = self._build_quality_gate(current_summary, alerts)
        return EvalTrendSummary(
            current_run_id=current_run_id,
            baseline_run_id=baseline_item.run_id if baseline_item else "",
            compared_runs=1 + (1 if baseline_item else 0) if current_run_id else (1 if baseline_item else 0),
            current_summary=current_summary,
            baseline_summary=baseline_summary,
            quality_gate=quality_gate,
            deltas=deltas,
            alerts=alerts,
        )

    def run(self, limit: int | None = None, persist: bool = True) -> EvalRunResult:
        started_at = utcnow()
        samples = self.dataset_store.list_samples()
        if limit is not None:
            samples = samples[:limit]

        case_results = [self._run_case(sample) for sample in samples]
        summary = self._summarize(case_results)
        finished_at = utcnow()
        run = EvalRunResult(
            run_id=f"eval_{uuid.uuid4().hex[:12]}",
            mode="offline",
            dataset_size=len(samples),
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            quality_gate=self._build_quality_gate(summary),
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
        primary_summary = self._summarize(primary_cases)
        shadow_summary = self._summarize(shadow_cases)
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
        winner, winner_reasons, primary_wins, shadow_wins, ties = self._select_shadow_winner(
            primary_summary,
            shadow_summary,
            diffs,
        )
        finished_at = utcnow()
        run = ShadowEvalRunResult(
            run_id=f"shadow_{uuid.uuid4().hex[:12]}",
            mode="shadow",
            dataset_size=len(samples),
            started_at=started_at,
            finished_at=finished_at,
            primary_summary=primary_summary,
            shadow_summary=shadow_summary,
            winner=winner,
            winner_reasons=winner_reasons,
            primary_wins=primary_wins,
            shadow_wins=shadow_wins,
            ties=ties,
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

    def _load_latest_run_payload(self, mode: str) -> dict | None:
        for item in self.run_store.list_runs(limit=50):
            if item.mode != mode:
                continue
            payload = self.run_store.load_run(item.run_id)
            if payload is not None:
                return payload
        return None

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

    @staticmethod
    def _build_quality_gate(
        summary: EvalRunSummary,
        alerts: list[EvalRegressionAlert] | None = None,
    ) -> EvalQualityGate:
        alerts = alerts or []
        if summary.total_cases == 0:
            return EvalQualityGate(
                status="warning",
                reasons=["No evaluation cases were executed."],
                blocking_metrics=[],
            )

        status = "pass"
        reasons: list[str] = []
        blocking_metrics: list[str] = []
        evidence_hit_rate = max(summary.retrieval_hit_rate, summary.title_hit_rate)

        if evidence_hit_rate < 0.8:
            status = "block"
            blocking_metrics.append("evidence_hit_rate")
            reasons.append("Evidence hit rate is below 0.80.")
        elif evidence_hit_rate < 0.9:
            status = "warning" if status == "pass" else status
            reasons.append("Evidence hit rate is below the target 0.90.")

        if summary.answer_valid_rate < 0.95:
            status = "block"
            blocking_metrics.append("answer_valid_rate")
            reasons.append("answer_valid_rate is below 0.95.")

        if summary.answer_match_rate < 0.75:
            status = "block"
            blocking_metrics.append("answer_match_rate")
            reasons.append("answer_match_rate is below 0.75.")
        elif summary.answer_match_rate < 0.85:
            status = "warning" if status == "pass" else status
            reasons.append("answer_match_rate is below the target 0.85.")

        if any(alert.severity == "critical" for alert in alerts):
            status = "block"
            blocking_metrics.extend(alert.metric for alert in alerts if alert.severity == "critical")
            reasons.append("Critical regression alerts were detected.")
        elif alerts and status == "pass":
            status = "warning"
            reasons.append("Regression alerts were detected.")

        return EvalQualityGate(
            status=status,
            reasons=list(dict.fromkeys(reasons)),
            blocking_metrics=list(dict.fromkeys(blocking_metrics)),
        )

    @staticmethod
    def _select_shadow_winner(
        primary_summary: EvalRunSummary,
        shadow_summary: EvalRunSummary,
        diffs: list[ShadowEvalCaseDiff],
    ) -> tuple[str, list[str], int, int, int]:
        primary_wins = 0
        shadow_wins = 0
        ties = 0
        reasons: list[str] = []
        metric_pairs = (
            ("retrieval_hit_rate", primary_summary.retrieval_hit_rate, shadow_summary.retrieval_hit_rate, True),
            ("title_hit_rate", primary_summary.title_hit_rate, shadow_summary.title_hit_rate, True),
            ("answer_match_rate", primary_summary.answer_match_rate, shadow_summary.answer_match_rate, True),
            ("answer_valid_rate", primary_summary.answer_valid_rate, shadow_summary.answer_valid_rate, True),
            ("average_latency_ms", primary_summary.average_latency_ms, shadow_summary.average_latency_ms, False),
        )
        for metric, primary_value, shadow_value, higher_is_better in metric_pairs:
            if round(primary_value, 3) == round(shadow_value, 3):
                ties += 1
                continue
            primary_better = primary_value > shadow_value if higher_is_better else primary_value < shadow_value
            if primary_better:
                primary_wins += 1
                reasons.append(f"primary leads on {metric}.")
            else:
                shadow_wins += 1
                reasons.append(f"shadow leads on {metric}.")

        changed_cases = sum(1 for diff in diffs if diff.changed)
        if changed_cases:
            reasons.append(f"{changed_cases} cases changed after shadow comparison.")

        if primary_wins > shadow_wins:
            winner = "primary"
        elif shadow_wins > primary_wins:
            winner = "shadow"
        else:
            winner = "tie"
            reasons.append("Primary and shadow are effectively tied on tracked metrics.")
        return winner, reasons[:6], primary_wins, shadow_wins, ties
