from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.deps import (
    get_document_ingestion_orchestrator,
    get_document_service,
    get_offline_evaluation_service,
)
from app.core.config import settings
from app.domain.auth.models import UserContext
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.evaluation.models import EvalDatasetImportRequest, EvalSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a minimal dataset and evaluation history so release-gate checks can run in CI or locally."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete local metadata, staged files, evaluation dataset, baseline, and run history before seeding.",
    )
    parser.add_argument(
        "--run-shadow",
        action="store_true",
        help="Run one shadow evaluation in addition to the offline evaluation.",
    )
    return parser.parse_args()


def build_release_gate_user() -> UserContext:
    """Return the internal admin identity used by release-gate fixture seeding."""

    return UserContext(
        user_id="release_gate_seed",
        tenant_id="release-gate",
        department_id="engineering",
        role="admin",
        clearance_level=5,
    )


def reset_local_state() -> None:
    """Remove local fixture files so release-gate preparation starts from a clean slate."""

    targets = [
        Path(settings.sqlite_path),
        Path(settings.eval_dataset_path),
        Path(settings.eval_baseline_path),
        Path(settings.eval_runs_dir),
        Path(settings.document_staging_dir),
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


def seed_documents() -> list[str]:
    """Create a deterministic knowledge fixture that should pass local offline evaluation."""

    user = build_release_gate_user()
    document_service = get_document_service()
    orchestrator = get_document_ingestion_orchestrator()
    seeded_doc_ids: list[str] = []
    payloads = [
        DocumentUploadRequest(
            title="报销制度",
            content="报销制度说明。\n\n审批时限为3个工作日。\n报销需附发票原件。",
            source_type="manual",
            department_scope=["engineering"],
            visibility_scope=["tenant"],
            security_level=1,
            tags=["finance", "release-gate"],
        ),
        DocumentUploadRequest(
            title="采购流程",
            content="采购流程说明。\n\n审批时限为2个工作日。\n采购需走部门审批。",
            source_type="manual",
            department_scope=["engineering"],
            visibility_scope=["tenant"],
            security_level=1,
            tags=["procurement", "release-gate"],
        ),
    ]
    for payload in payloads:
        document = document_service.upload_document(payload, user)
        document = orchestrator.process_document(document.id)
        seeded_doc_ids.append(document.id)
    return seeded_doc_ids


def seed_evaluation_data() -> dict:
    """Persist reviewed evaluation samples, a baseline, and a fresh offline run."""

    user = build_release_gate_user()
    service = get_offline_evaluation_service()
    samples = [
        EvalSample(
            id="release_gate_case_finance",
            query="报销审批时限是什么？",
            scene="standard_qa",
            expected_titles=["报销制度"],
            expected_answer_contains=["3个工作日"],
            tenant_id=user.tenant_id,
            department_id=user.department_id,
            role=user.role,
            clearance_level=user.clearance_level,
            labels=["golden", "release-gate"],
            reviewed=True,
            reviewed_by="release_gate_seed",
            notes="Release gate seed case for finance policy coverage.",
        ),
        EvalSample(
            id="release_gate_case_procurement",
            query="采购审批时限是什么？",
            scene="standard_qa",
            expected_titles=["采购流程"],
            expected_answer_contains=["2个工作日"],
            tenant_id=user.tenant_id,
            department_id=user.department_id,
            role=user.role,
            clearance_level=user.clearance_level,
            labels=["golden", "release-gate"],
            reviewed=True,
            reviewed_by="release_gate_seed",
            notes="Release gate seed case for procurement process coverage.",
        ),
    ]
    service.import_samples(EvalDatasetImportRequest(mode="replace", samples=samples))

    baseline = service.get_quality_baseline()
    baseline.name = "CI Release Gate Baseline"
    baseline.minimum_review_coverage = 1.0
    baseline.require_shadow_run = True
    baseline.shadow_must_not_lose = True
    service.update_quality_baseline(baseline)

    offline_run = service.run()
    return {
        "offline_run_id": offline_run.run_id,
        "offline_quality_gate": offline_run.quality_gate.status,
        "dataset_size": offline_run.dataset_size,
    }


def main() -> int:
    args = parse_args()
    if args.reset:
        reset_local_state()

    summary = {"seeded_doc_ids": seed_documents()}
    summary.update(seed_evaluation_data())

    if args.run_shadow:
        shadow_run = get_offline_evaluation_service().run_shadow()
        summary.update(
            {
                "shadow_run_id": shadow_run.run_id,
                "shadow_winner": shadow_run.winner,
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
