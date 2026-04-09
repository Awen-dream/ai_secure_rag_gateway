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

    def get_sample(self, sample_id: str) -> EvalSample | None:
        for sample in self.list_samples():
            if sample.id == sample_id:
                return sample
        return None

    def upsert_sample(self, sample: EvalSample) -> EvalSample:
        samples = self.list_samples()
        updated = False
        for index, existing in enumerate(samples):
            if existing.id == sample.id:
                samples[index] = sample
                updated = True
                break
        if not updated:
            samples.append(sample)
        self.replace_samples(samples)
        return sample

    def delete_sample(self, sample_id: str) -> bool:
        samples = self.list_samples()
        filtered = [sample for sample in samples if sample.id != sample_id]
        if len(filtered) == len(samples):
            return False
        self.replace_samples(filtered)
        return True
