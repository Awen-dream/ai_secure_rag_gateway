from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime

from app.application.context.builder import ContextBuilderService
from app.application.evaluation.engines import EvaluationExecutionEngine, NativeEvaluationExecutionEngine
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.domain.evaluation.models import (
    EvalBulkAnnotationRequest,
    EvalBulkAnnotationResult,
    EvalCaseResult,
    EvalDatasetExport,
    EvalDatasetImportRequest,
    EvalDatasetImportResult,
    EvalDatasetStats,
    EvalQualityBaseline,
    EvalQualityGate,
    EvalRegressionAlert,
    EvalRunListItem,
    EvalRunResult,
    EvalRunSummary,
    EvalSampleTemplate,
    EvalTrendSummary,
    EvalSample,
    ReleaseGateCheck,
    ReleaseGateReport,
    ReleaseReadinessReport,
    ShadowReportSummary,
    ShadowEvalCaseDiff,
    ShadowEvalRunResult,
)
from app.domain.retrieval.services import RetrievalService
from app.infrastructure.storage.local_eval_baseline_store import LocalEvalBaselineStore
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore
from app.infrastructure.storage.local_eval_run_store import LocalEvalRunStore


def utcnow() -> datetime:
    return datetime.utcnow()


class OfflineEvaluationService:
    """Run offline evaluation over the current retrieval and generation stack."""

    def __init__(
        self,
        dataset_store: LocalEvalDatasetStore,
        baseline_store: LocalEvalBaselineStore,
        run_store: LocalEvalRunStore,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
        execution_engine: EvaluationExecutionEngine | None = None,
    ) -> None:
        self.dataset_store = dataset_store
        self.baseline_store = baseline_store
        self.run_store = run_store
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.generation_service = generation_service
        self.execution_engine = execution_engine or NativeEvaluationExecutionEngine()

    def list_samples(self) -> list[EvalSample]:
        return self.dataset_store.list_samples()

    def build_sample_template(self, scene: str = "standard_qa") -> EvalSampleTemplate:
        """Return one starter sample template and batch example for dataset authoring workflows."""

        template = EvalSample(
            id="sample_xxx",
            query="报销审批时限是什么？",
            scene=scene,
            expected_doc_ids=["doc_xxx"],
            expected_titles=["报销制度"],
            expected_answer_contains=["3个工作日"],
            expected_intent="standard_qa",
            labels=["finance", "golden"],
            reviewed=False,
            reviewed_by="",
            notes="补充预期证据与答案关键字后即可纳入离线评测。",
            metadata={"owner": "qa", "priority": "high"},
        )
        return EvalSampleTemplate(
            scene=scene,
            sample=template,
            batch_example=[
                template,
                template.model_copy(
                    update={
                        "id": "sample_policy_2",
                        "query": "采购审批流程怎么走？",
                        "expected_titles": ["采购流程"],
                        "expected_answer_contains": ["审批流程"],
                        "labels": ["process"],
                    }
                ),
            ],
        )

    def get_sample(self, sample_id: str) -> EvalSample | None:
        return self.dataset_store.get_sample(sample_id)

    def upsert_sample(self, sample: EvalSample) -> EvalSample:
        return self.dataset_store.upsert_sample(sample)

    def delete_sample(self, sample_id: str) -> bool:
        return self.dataset_store.delete_sample(sample_id)

    def bulk_annotate(self, payload: EvalBulkAnnotationRequest) -> EvalBulkAnnotationResult:
        samples = self.dataset_store.list_samples()
        updated_ids: list[str] = []
        for sample in samples:
            if sample.id not in set(payload.sample_ids):
                continue
            if payload.labels:
                if payload.replace_labels:
                    sample.labels = list(dict.fromkeys(payload.labels))
                else:
                    sample.labels = list(dict.fromkeys([*sample.labels, *payload.labels]))
            if payload.status is not None:
                sample.status = payload.status
            if payload.reviewed is not None:
                sample.reviewed = payload.reviewed
            if payload.reviewed_by.strip():
                sample.reviewed_by = payload.reviewed_by.strip()
            if payload.notes.strip():
                sample.notes = payload.notes.strip()
            updated_ids.append(sample.id)
        self.dataset_store.replace_samples(samples)
        return EvalBulkAnnotationResult(updated_count=len(updated_ids), updated_ids=updated_ids)

    def import_samples(self, payload: EvalDatasetImportRequest) -> EvalDatasetImportResult:
        """Import evaluation samples using replace or upsert semantics."""

        mode = payload.mode.strip().lower() or "replace"
        if mode == "replace":
            count = self.dataset_store.replace_samples(payload.samples)
            return EvalDatasetImportResult(mode=mode, sample_count=count, created_count=count, updated_count=0)

        existing = {sample.id: sample for sample in self.dataset_store.list_samples()}
        created_count = 0
        updated_count = 0
        for sample in payload.samples:
            if sample.id in existing:
                updated_count += 1
            else:
                created_count += 1
            existing[sample.id] = sample
        merged = list(existing.values())
        self.dataset_store.replace_samples(merged)
        return EvalDatasetImportResult(
            mode=mode,
            sample_count=len(merged),
            created_count=created_count,
            updated_count=updated_count,
        )

    def export_samples(self, export_format: str = "json") -> EvalDatasetExport:
        """Return evaluation samples as JSON payload plus JSONL text for external export."""

        samples = self.dataset_store.list_samples()
        jsonl = "\n".join(sample.model_dump_json() for sample in samples)
        if jsonl:
            jsonl += "\n"
        return EvalDatasetExport(
            format=export_format,
            sample_count=len(samples),
            samples=samples,
            jsonl=jsonl,
        )

    def dataset_stats(self) -> EvalDatasetStats:
        samples = self.dataset_store.list_samples()
        if not samples:
            return EvalDatasetStats()
        scene_counts = Counter(sample.scene for sample in samples)
        label_counts = Counter(label for sample in samples for label in sample.labels)
        status_counts = Counter(sample.status for sample in samples)
        reviewed = sum(1 for sample in samples if sample.reviewed)
        active = sum(1 for sample in samples if sample.status == "active")
        return EvalDatasetStats(
            total_samples=len(samples),
            active_samples=active,
            reviewed_samples=reviewed,
            coverage_rate=round(reviewed / len(samples), 3),
            scenes=dict(scene_counts),
            labels=dict(label_counts),
            statuses=dict(status_counts),
        )

    def get_quality_baseline(self) -> EvalQualityBaseline:
        baseline = self.baseline_store.load()
        if baseline is None:
            baseline = EvalQualityBaseline()
            self.baseline_store.save(baseline)
        return baseline

    def update_quality_baseline(self, baseline: EvalQualityBaseline) -> EvalQualityBaseline:
        baseline.updated_at = utcnow()
        if not baseline.created_at:
            baseline.created_at = baseline.updated_at
        return self.baseline_store.save(baseline)

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
        baseline = self.get_quality_baseline()
        dataset_stats = self.dataset_stats()
        latest_offline_payload = self._load_latest_run_payload(mode="offline")
        if latest_offline_payload is None:
            return ReleaseReadinessReport(
                generated_at=utcnow(),
                decision="hold",
                baseline=baseline,
                dataset_stats=dataset_stats,
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

        if baseline.require_shadow_run and baseline.shadow_must_not_lose and shadow_report.winner == "shadow":
            if decision == "ready":
                decision = "review"
            reasons.append("Shadow baseline currently outperforms the primary stack.")
        elif baseline.require_shadow_run and shadow_report.winner == "unavailable":
            if decision == "ready":
                decision = "review"
            reasons.append("No shadow evaluation run is available.")

        if dataset_stats.coverage_rate < baseline.minimum_review_coverage:
            if decision == "ready":
                decision = "review"
            reasons.append(
                f"Evaluation review coverage {dataset_stats.coverage_rate:.3f} is below minimum {baseline.minimum_review_coverage:.3f}."
            )

        if not reasons:
            reasons.append("Offline evaluation and shadow comparison both meet release expectations.")

        return ReleaseReadinessReport(
            generated_at=utcnow(),
            decision=decision,
            latest_offline_run_id=latest_offline_run.run_id,
            latest_shadow_run_id=shadow_report.latest_run_id,
            baseline=baseline,
            dataset_stats=dataset_stats,
            quality_gate=latest_offline_run.quality_gate,
            trend=trend,
            shadow_report=shadow_report,
            reasons=reasons,
        )

    def build_release_gate_report(self, allow_review: bool = False) -> ReleaseGateReport:
        """Build one release gate report with explicit checklist items and pass/fail state."""

        readiness = self.build_release_readiness_report()
        checks: list[ReleaseGateCheck] = []
        dataset_stats = readiness.dataset_stats
        baseline = readiness.baseline

        checks.append(
            ReleaseGateCheck(
                name="dataset_presence",
                status="pass" if dataset_stats.total_samples > 0 else "block",
                severity="critical" if dataset_stats.total_samples == 0 else "info",
                detail=f"evaluation samples: {dataset_stats.total_samples}",
            )
        )
        checks.append(
            ReleaseGateCheck(
                name="review_coverage",
                status="pass" if dataset_stats.coverage_rate >= baseline.minimum_review_coverage else "warning",
                severity="warning",
                detail=(
                    f"coverage={dataset_stats.coverage_rate:.3f}, minimum={baseline.minimum_review_coverage:.3f}"
                ),
            )
        )
        checks.append(
            ReleaseGateCheck(
                name="quality_gate",
                status=readiness.quality_gate.status,
                severity="critical" if readiness.quality_gate.status == "block" else "warning",
                detail="; ".join(readiness.quality_gate.reasons) or "quality gate passed",
            )
        )
        checks.append(
            ReleaseGateCheck(
                name="shadow_report",
                status="pass"
                if not baseline.require_shadow_run
                or readiness.shadow_report.winner in {"primary", "tie"}
                else ("warning" if readiness.shadow_report.winner == "unavailable" else "block"),
                severity="warning",
                detail=(
                    f"winner={readiness.shadow_report.winner}, recommendation={readiness.shadow_report.recommendation}"
                ),
            )
        )
        checks.append(
            ReleaseGateCheck(
                name="release_decision",
                status=readiness.decision,
                severity="critical" if readiness.decision == "hold" else "warning",
                detail="; ".join(readiness.reasons),
            )
        )

        passed = readiness.decision == "ready" or (allow_review and readiness.decision == "review")
        return ReleaseGateReport(
            generated_at=utcnow(),
            decision=readiness.decision,
            passed=passed,
            allow_review=allow_review,
            checks=checks,
            release_readiness=readiness,
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
        baseline = self.get_quality_baseline()
        alerts = self._build_regression_alerts(current_summary, baseline_summary, deltas, baseline)
        quality_gate = self._build_quality_gate(current_summary, alerts, baseline=baseline)
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
        samples = [sample for sample in self.dataset_store.list_samples() if sample.status != "archived"]
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
            quality_gate=self._build_quality_gate(summary, baseline=self.get_quality_baseline()),
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
        return self.execution_engine.run_case(
            sample=sample,
            retrieval_service=retrieval_service or self.retrieval_service,
            context_builder=self.context_builder,
            prompt_builder=self.prompt_builder,
            generation_service=self.generation_service,
        )

    def _build_shadow_retrieval_service(self) -> RetrievalService:
        return self.execution_engine.build_shadow_retrieval_service(self.retrieval_service)

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
        baseline: EvalQualityBaseline,
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
            if delta <= -baseline.regression_warning_drop:
                current_value = float(getattr(current_summary, metric))
                baseline_value = float(getattr(baseline_summary, metric))
                alerts.append(
                    EvalRegressionAlert(
                        metric=metric,
                        severity="critical" if delta <= -baseline.regression_block_drop else "warning",
                        direction="down",
                        current_value=current_value,
                        baseline_value=baseline_value,
                        delta=delta,
                        message=f"{metric} regressed from {baseline_value:.3f} to {current_value:.3f}.",
                    )
                )

        latency_delta = float(deltas.get("average_latency_ms", 0.0))
        if (
            latency_delta >= baseline.latency_warning_increase_ms
            and baseline_summary.average_latency_ms > 0
            and current_summary.average_latency_ms >= baseline_summary.average_latency_ms * baseline.latency_warning_multiplier
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
        baseline: EvalQualityBaseline | None = None,
    ) -> EvalQualityGate:
        alerts = alerts or []
        baseline = baseline or EvalQualityBaseline()
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

        if evidence_hit_rate < baseline.min_evidence_hit_rate:
            status = "block"
            blocking_metrics.append("evidence_hit_rate")
            reasons.append(f"Evidence hit rate is below {baseline.min_evidence_hit_rate:.2f}.")
        elif evidence_hit_rate < baseline.target_evidence_hit_rate:
            status = "warning" if status == "pass" else status
            reasons.append(f"Evidence hit rate is below the target {baseline.target_evidence_hit_rate:.2f}.")

        if summary.answer_valid_rate < baseline.min_answer_valid_rate:
            status = "block"
            blocking_metrics.append("answer_valid_rate")
            reasons.append(f"answer_valid_rate is below {baseline.min_answer_valid_rate:.2f}.")

        if summary.answer_match_rate < baseline.min_answer_match_rate:
            status = "block"
            blocking_metrics.append("answer_match_rate")
            reasons.append(f"answer_match_rate is below {baseline.min_answer_match_rate:.2f}.")
        elif summary.answer_match_rate < baseline.target_answer_match_rate:
            status = "warning" if status == "pass" else status
            reasons.append(f"answer_match_rate is below the target {baseline.target_answer_match_rate:.2f}.")

        if summary.average_latency_ms > baseline.max_latency_ms:
            status = "warning" if status == "pass" else status
            reasons.append(f"average_latency_ms exceeds baseline max {baseline.max_latency_ms:.0f} ms.")

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
