from __future__ import annotations

import json
import sys
from pathlib import Path

from app.core.config import settings
from app.domain.evaluation.models import EvalSample
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore


def _load_input(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list of samples.")
        return payload
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/load_eval_dataset.py <dataset.json|dataset.jsonl>")
        return 1

    source_path = Path(sys.argv[1])
    if not source_path.exists():
        print(f"Dataset file not found: {source_path}")
        return 1

    payload = _load_input(source_path)
    samples = [EvalSample.model_validate(item) for item in payload]
    store = LocalEvalDatasetStore(settings.eval_dataset_path)
    count = store.replace_samples(samples)
    print(f"Loaded {count} evaluation samples into {settings.eval_dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
