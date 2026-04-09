from __future__ import annotations

import json
from pathlib import Path

from app.domain.evaluation.models import EvalSample


class LocalEvalDatasetStore:
    """Persist evaluation samples as newline-delimited JSON for local/offline evaluation."""

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def list_samples(self) -> list[EvalSample]:
        if not self.dataset_path.exists():
            return []
        samples: list[EvalSample] = []
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            samples.append(EvalSample.model_validate(json.loads(stripped)))
        return samples

    def replace_samples(self, samples: list[EvalSample]) -> int:
        payload = "\n".join(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) for sample in samples)
        if payload:
            payload += "\n"
        self.dataset_path.write_text(payload, encoding="utf-8")
        return len(samples)
