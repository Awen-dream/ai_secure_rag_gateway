import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class ReleaseGateScriptsTest(unittest.TestCase):
    def test_prepare_fixture_allows_release_gate_to_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="secure_rag_release_gate_") as tmpdir:
            tmp = Path(tmpdir)
            env = os.environ.copy()
            env.update(
                {
                    "APP_REPOSITORY_BACKEND": "sqlite",
                    "APP_SQLITE_PATH": str(tmp / "release-gate.db"),
                    "APP_REDIS_MODE": "local-fallback",
                    "APP_ELASTICSEARCH_MODE": "local-fallback",
                    "APP_PGVECTOR_MODE": "local-fallback",
                    "APP_EVAL_DATASET_PATH": str(tmp / "evaluation" / "dataset.jsonl"),
                    "APP_EVAL_RUNS_DIR": str(tmp / "evaluation" / "runs"),
                    "APP_EVAL_BASELINE_PATH": str(tmp / "evaluation" / "baseline.json"),
                    "APP_DOCUMENT_STAGING_DIR": str(tmp / "staging" / "documents"),
                    "OPENAI_API_KEY": "",
                    "PYTHONPATH": ".",
                }
            )

            prepare = subprocess.run(
                [sys.executable, "scripts/prepare_release_gate_fixture.py", "--reset", "--run-shadow"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stderr)
            prepare_payload = json.loads(prepare.stdout)
            self.assertEqual(prepare_payload["offline_quality_gate"], "pass")
            self.assertIn("shadow_run_id", prepare_payload)

            gate = subprocess.run(
                [sys.executable, "scripts/run_release_gate.py", "--json"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(gate.returncode, 0, gate.stdout)
            gate_payload = json.loads(gate.stdout)
            self.assertTrue(gate_payload["passed"])
            self.assertEqual(gate_payload["decision"], "ready")


if __name__ == "__main__":
    unittest.main()
