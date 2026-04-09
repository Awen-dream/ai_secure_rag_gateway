from __future__ import annotations

import json
from pathlib import Path

from app.domain.evaluation.models import EvalQualityBaseline


class LocalEvalBaselineStore:
    """Persist one evaluation quality baseline as JSON for release-gate and quality-gate decisions."""

    def __init__(self, baseline_path: str) -> None:
        self.baseline_path = Path(baseline_path)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> EvalQualityBaseline | None:
        if not self.baseline_path.exists():
            return None
        payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        return EvalQualityBaseline.model_validate(payload)

    def save(self, baseline: EvalQualityBaseline) -> EvalQualityBaseline:
        self.baseline_path.write_text(
            json.dumps(baseline.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return baseline
