from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.deps import get_offline_evaluation_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evaluation release gate and exit non-zero on failure.")
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Treat release decision=review as pass for the purpose of this gate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full release gate report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = get_offline_evaluation_service()
    report = service.build_release_gate_report(allow_review=args.allow_review)

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(f"decision={report.decision} passed={report.passed} allow_review={report.allow_review}")
        for check in report.checks:
            print(f"- {check.name}: {check.status} ({check.severity}) {check.detail}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
