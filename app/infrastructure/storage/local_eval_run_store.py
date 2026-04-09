from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.domain.evaluation.models import EvalRunListItem


class LocalEvalRunStore:
    """Persist offline/shadow evaluation runs as JSON files for later inspection."""

    def __init__(self, runs_dir: str) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, payload: dict) -> str:
        path = self.runs_dir / f"{run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def load_run(self, run_id: str) -> dict | None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self, limit: int = 20) -> list[EvalRunListItem]:
        payloads: list[dict] = []
        for path in self.runs_dir.glob("*.json"):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))

        payloads.sort(
            key=lambda item: _parse_datetime(item.get("started_at")),
            reverse=True,
        )

        items: list[EvalRunListItem] = []
        for payload in payloads:
            items.append(
                EvalRunListItem(
                    run_id=payload.get("run_id", ""),
                    mode=payload.get("mode", "offline"),
                    dataset_size=payload.get("dataset_size", 0),
                    started_at=payload["started_at"],
                    finished_at=payload["finished_at"],
                    summary=payload.get("summary")
                    or {
                        "primary_summary": payload.get("primary_summary", {}),
                        "shadow_summary": payload.get("shadow_summary", {}),
                    },
                )
            )
            if len(items) >= limit:
                break
        return items


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min
