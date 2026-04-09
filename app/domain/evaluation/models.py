from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    id: str
    query: str
    scene: str = "standard_qa"
    status: str = "active"
    reviewed: bool = False
    reviewed_by: str = ""
    labels: list[str] = Field(default_factory=list)
    notes: str = ""
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_titles: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_intent: Optional[str] = None
    tenant_id: str = "eval"
    user_id: str = "eval_user"
    department_id: str = "engineering"
    role: str = "admin"
    clearance_level: int = 3
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    sample_id: str
    query: str
    scene: str
    hit_expected_doc: bool = False
    hit_expected_title: bool = False
    answer_contains_expected: bool = False
    answer_valid: bool = False
    matched_doc_ids: list[str] = Field(default_factory=list)
    matched_titles: list[str] = Field(default_factory=list)
    retrieved_chunks: int = 0
    rewritten_query: str = ""
    intent: str = ""
    latency_ms: int = 0
    validation_missing_sections: list[str] = Field(default_factory=list)
    answer_preview: str = ""
    citations: list[str] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    total_cases: int = 0
    retrieval_hit_rate: float = 0.0
    title_hit_rate: float = 0.0
    answer_match_rate: float = 0.0
    answer_valid_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_retrieved_chunks: float = 0.0


class EvalDatasetStats(BaseModel):
    total_samples: int = 0
    active_samples: int = 0
    reviewed_samples: int = 0
    coverage_rate: float = 0.0
    scenes: dict[str, int] = Field(default_factory=dict)
    labels: dict[str, int] = Field(default_factory=dict)
    statuses: dict[str, int] = Field(default_factory=dict)


class EvalDatasetImportRequest(BaseModel):
    mode: str = "replace"
    samples: list[EvalSample] = Field(default_factory=list)


class EvalDatasetImportResult(BaseModel):
    mode: str = "replace"
    sample_count: int = 0
    created_count: int = 0
    updated_count: int = 0


class EvalDatasetExport(BaseModel):
    format: str = "json"
    sample_count: int = 0
    samples: list[EvalSample] = Field(default_factory=list)
    jsonl: str = ""


class EvalSampleTemplate(BaseModel):
    scene: str = "standard_qa"
    sample: EvalSample
    batch_example: list[EvalSample] = Field(default_factory=list)


class EvalBulkAnnotationRequest(BaseModel):
    sample_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    status: Optional[str] = None
    reviewed: Optional[bool] = None
    reviewed_by: str = ""
    notes: str = ""
    replace_labels: bool = False


class EvalBulkAnnotationResult(BaseModel):
    updated_count: int = 0
    updated_ids: list[str] = Field(default_factory=list)


class EvalQualityBaseline(BaseModel):
    id: str = "default"
    name: str = "Default Quality Baseline"
    min_evidence_hit_rate: float = 0.8
    target_evidence_hit_rate: float = 0.9
    min_answer_valid_rate: float = 0.95
    min_answer_match_rate: float = 0.75
    target_answer_match_rate: float = 0.85
    regression_warning_drop: float = 0.05
    regression_block_drop: float = 0.1
    max_latency_ms: float = 1200.0
    latency_warning_increase_ms: float = 100.0
    latency_warning_multiplier: float = 1.2
    require_shadow_run: bool = True
    shadow_must_not_lose: bool = True
    minimum_review_coverage: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EvalQualityGate(BaseModel):
    status: str = "pass"
    reasons: list[str] = Field(default_factory=list)
    blocking_metrics: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    run_id: str = ""
    mode: str = "offline"
    dataset_size: int
    started_at: datetime
    finished_at: datetime
    summary: EvalRunSummary
    quality_gate: EvalQualityGate = Field(default_factory=EvalQualityGate)
    cases: list[EvalCaseResult] = Field(default_factory=list)


class ShadowEvalCaseDiff(BaseModel):
    sample_id: str
    query: str
    primary_hit: bool = False
    shadow_hit: bool = False
    primary_answer_match: bool = False
    shadow_answer_match: bool = False
    changed: bool = False
    primary_rewritten_query: str = ""
    shadow_rewritten_query: str = ""


class ShadowEvalRunResult(BaseModel):
    run_id: str = ""
    mode: str = "shadow"
    dataset_size: int
    started_at: datetime
    finished_at: datetime
    primary_summary: EvalRunSummary
    shadow_summary: EvalRunSummary
    winner: str = "tie"
    winner_reasons: list[str] = Field(default_factory=list)
    primary_wins: int = 0
    shadow_wins: int = 0
    ties: int = 0
    diffs: list[ShadowEvalCaseDiff] = Field(default_factory=list)


class EvalRunListItem(BaseModel):
    run_id: str
    mode: str
    dataset_size: int
    started_at: datetime
    finished_at: datetime
    summary: dict = Field(default_factory=dict)


class EvalRegressionAlert(BaseModel):
    metric: str
    severity: str = "warning"
    direction: str
    current_value: float
    baseline_value: float
    delta: float
    message: str


class EvalTrendSummary(BaseModel):
    current_run_id: str = ""
    baseline_run_id: str = ""
    compared_runs: int = 0
    current_summary: EvalRunSummary = Field(default_factory=EvalRunSummary)
    baseline_summary: Optional[EvalRunSummary] = None
    quality_gate: EvalQualityGate = Field(default_factory=EvalQualityGate)
    deltas: dict = Field(default_factory=dict)
    alerts: list[EvalRegressionAlert] = Field(default_factory=list)


class ShadowReportSummary(BaseModel):
    latest_run_id: str = ""
    winner: str = "unavailable"
    recommendation: str = "unavailable"
    primary_wins: int = 0
    shadow_wins: int = 0
    ties: int = 0
    changed_cases: int = 0
    winner_reasons: list[str] = Field(default_factory=list)


class ReleaseReadinessReport(BaseModel):
    generated_at: datetime
    decision: str = "hold"
    latest_offline_run_id: str = ""
    latest_shadow_run_id: str = ""
    baseline: EvalQualityBaseline = Field(default_factory=EvalQualityBaseline)
    dataset_stats: EvalDatasetStats = Field(default_factory=EvalDatasetStats)
    quality_gate: EvalQualityGate = Field(default_factory=EvalQualityGate)
    trend: EvalTrendSummary = Field(default_factory=EvalTrendSummary)
    shadow_report: ShadowReportSummary = Field(default_factory=ShadowReportSummary)
    reasons: list[str] = Field(default_factory=list)


class ReleaseGateCheck(BaseModel):
    name: str
    status: str = "pass"
    severity: str = "info"
    detail: str = ""


class ReleaseGateReport(BaseModel):
    generated_at: datetime
    decision: str = "hold"
    passed: bool = False
    allow_review: bool = False
    checks: list[ReleaseGateCheck] = Field(default_factory=list)
    release_readiness: ReleaseReadinessReport
