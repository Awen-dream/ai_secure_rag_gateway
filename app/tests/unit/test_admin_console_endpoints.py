import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient


class AdminConsoleEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_admin_console.db"
        self.eval_dataset_path = "/tmp/secure_rag_gateway_admin_console_dataset.jsonl"
        self.eval_runs_dir = "/tmp/secure_rag_gateway_admin_console_runs"
        self.eval_baseline_path = "/tmp/secure_rag_gateway_admin_console_baseline.json"
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.eval_dataset_path).unlink(missing_ok=True)
        Path(self.eval_baseline_path).unlink(missing_ok=True)
        eval_runs = Path(self.eval_runs_dir)
        if eval_runs.exists():
            for path in eval_runs.glob("*.json"):
                path.unlink()

        import os

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_EVAL_DATASET_PATH"] = self.eval_dataset_path
        os.environ["APP_EVAL_RUNS_DIR"] = self.eval_runs_dir
        os.environ["APP_EVAL_BASELINE_PATH"] = self.eval_baseline_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.eval_dataset_path = self.eval_dataset_path
        settings.eval_runs_dir = self.eval_runs_dir
        settings.eval_baseline_path = self.eval_baseline_path
        settings.redis_mode = "local-fallback"
        settings.openai_api_key = None

        from app.api.deps import (
            get_admin_console_service,
            get_audit_service,
            get_chat_service,
            get_context_builder_service,
            get_document_ingestion_orchestrator,
            get_document_ingestion_worker,
            get_document_service,
            get_document_source_store,
            get_document_task_queue,
            get_embedding_client,
            get_eval_baseline_store,
            get_eval_dataset_store,
            get_eval_run_store,
            get_feishu_client,
            get_feishu_source_sync_service,
            get_generation_service,
            get_indexing_service,
            get_keyword_backend,
            get_offline_evaluation_service,
            get_openai_client,
            get_output_guard,
            get_policy_engine,
            get_prompt_builder_service,
            get_prompt_template_service,
            get_query_planning_service,
            get_query_understanding_service,
            get_rate_limit_service,
            get_recall_planning_service,
            get_redis_client,
            get_repository,
            get_retrieval_cache,
            get_retrieval_rerank_service,
            get_retrieval_reranker,
            get_retrieval_service,
            get_session_cache,
            get_vector_backend,
        )

        for factory in (
            get_repository,
            get_redis_client,
            get_session_cache,
            get_retrieval_cache,
            get_rate_limit_service,
            get_document_task_queue,
            get_embedding_client,
            get_feishu_client,
            get_feishu_source_sync_service,
            get_keyword_backend,
            get_vector_backend,
            get_document_source_store,
            get_indexing_service,
            get_document_ingestion_orchestrator,
            get_document_ingestion_worker,
            get_document_service,
            get_prompt_template_service,
            get_prompt_builder_service,
            get_policy_engine,
            get_output_guard,
            get_audit_service,
            get_openai_client,
            get_query_understanding_service,
            get_query_planning_service,
            get_recall_planning_service,
            get_retrieval_reranker,
            get_retrieval_rerank_service,
            get_context_builder_service,
            get_generation_service,
            get_eval_dataset_store,
            get_eval_baseline_store,
            get_eval_run_store,
            get_offline_evaluation_service,
            get_admin_console_service,
            get_retrieval_service,
            get_chat_service,
        ):
            factory.cache_clear()

        from app.main import app

        self.client = TestClient(app)
        self.headers = {
            "X-User-Id": "u1",
            "X-Tenant-Id": "t1",
            "X-Department-Id": "engineering",
            "X-Role": "admin",
            "X-Clearance-Level": "5",
        }

    def test_admin_dashboard_summary_and_documents_inventory(self) -> None:
        self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "报销制度",
                "content": "报销制度说明。\n\n审批时限为3个工作日。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )

        dashboard = self.client.get("/api/v1/admin/dashboard/summary", headers=self.headers)
        self.assertEqual(dashboard.status_code, 200)
        payload = dashboard.json()
        self.assertEqual(payload["documents"]["total"], 1)
        self.assertGreaterEqual(payload["traffic"]["total_queries"], 1)
        self.assertIn("release_readiness", payload["evaluation"])

        documents = self.client.get("/api/v1/admin/documents", headers=self.headers)
        self.assertEqual(documents.status_code, 200)
        docs_payload = documents.json()
        self.assertEqual(len(docs_payload), 1)
        self.assertEqual(docs_payload[0]["title"], "报销制度")
        self.assertGreaterEqual(docs_payload[0]["chunk_count"], 1)

    def test_admin_can_manage_evaluation_dataset_and_console_page(self) -> None:
        self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "采购流程",
                "content": "采购流程说明。\n\n审批时限为2个工作日。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        bootstrap = self.client.post("/api/v1/admin/evaluation/dataset/bootstrap?limit=10", headers=self.headers)
        self.assertEqual(bootstrap.status_code, 200)
        bootstrap_payload = bootstrap.json()
        self.assertGreaterEqual(bootstrap_payload["sample_count"], 1)

        dataset = self.client.get("/api/v1/admin/evaluation/dataset", headers=self.headers)
        self.assertEqual(dataset.status_code, 200)
        self.assertGreaterEqual(len(dataset.json()), 1)

        replacement = self.client.post(
            "/api/v1/admin/evaluation/dataset/replace",
            content=json.dumps(
                [
                    {
                        "id": "sample_custom_1",
                        "query": "采购审批时限是什么？",
                        "scene": "standard_qa",
                        "expected_titles": ["采购流程"],
                        "expected_answer_contains": ["2个工作日"],
                    }
                ]
            ),
            headers={**self.headers, "Content-Type": "application/json"},
        )
        self.assertEqual(replacement.status_code, 200)
        self.assertEqual(replacement.json()["sample_count"], 1)

        console = self.client.get("/admin-console")
        self.assertEqual(console.status_code, 200)
        self.assertIn("Secure Enterprise RAG Gateway Admin Console", console.text)

    def test_admin_can_bulk_annotate_samples_and_update_quality_baseline(self) -> None:
        self.client.post(
            "/api/v1/admin/evaluation/dataset/replace",
            content=json.dumps(
                [
                    {
                        "id": "sample_1",
                        "query": "报销审批时限是什么？",
                        "scene": "standard_qa",
                        "expected_titles": ["报销制度"],
                    },
                    {
                        "id": "sample_2",
                        "query": "采购审批时限是什么？",
                        "scene": "standard_qa",
                        "expected_titles": ["采购流程"],
                    },
                ]
            ),
            headers={**self.headers, "Content-Type": "application/json"},
        )

        annotate = self.client.post(
            "/api/v1/admin/evaluation/dataset/bulk-annotate",
            json={
                "sample_ids": ["sample_1", "sample_2"],
                "labels": ["finance", "reviewed"],
                "reviewed": True,
                "reviewed_by": "qa_admin",
                "notes": "批量完成首轮标注",
            },
            headers=self.headers,
        )
        self.assertEqual(annotate.status_code, 200)
        self.assertEqual(annotate.json()["updated_count"], 2)

        stats = self.client.get("/api/v1/admin/evaluation/dataset/stats", headers=self.headers)
        self.assertEqual(stats.status_code, 200)
        stats_payload = stats.json()
        self.assertEqual(stats_payload["reviewed_samples"], 2)
        self.assertEqual(stats_payload["coverage_rate"], 1.0)
        self.assertEqual(stats_payload["labels"]["finance"], 2)

        baseline_update = self.client.post(
            "/api/v1/admin/evaluation/baseline",
            json={
                "id": "default",
                "name": "Strict Gate",
                "min_evidence_hit_rate": 0.95,
                "target_evidence_hit_rate": 0.98,
                "min_answer_valid_rate": 0.99,
                "min_answer_match_rate": 0.9,
                "target_answer_match_rate": 0.95,
                "regression_warning_drop": 0.03,
                "regression_block_drop": 0.08,
                "max_latency_ms": 800.0,
                "latency_warning_increase_ms": 80.0,
                "latency_warning_multiplier": 1.15,
                "require_shadow_run": True,
                "shadow_must_not_lose": True,
                "minimum_review_coverage": 1.0,
            },
            headers=self.headers,
        )
        self.assertEqual(baseline_update.status_code, 200)
        self.assertEqual(baseline_update.json()["name"], "Strict Gate")

        overview = self.client.get("/api/v1/admin/evaluation/dataset/overview", headers=self.headers)
        self.assertEqual(overview.status_code, 200)
        overview_payload = overview.json()
        self.assertEqual(overview_payload["stats"]["reviewed_samples"], 2)
        self.assertEqual(overview_payload["baseline"]["name"], "Strict Gate")

    def test_admin_can_import_export_template_and_release_gate(self) -> None:
        template = self.client.get("/api/v1/admin/evaluation/dataset/template?scene=standard_qa", headers=self.headers)
        self.assertEqual(template.status_code, 200)
        template_payload = template.json()
        self.assertEqual(template_payload["scene"], "standard_qa")
        self.assertGreaterEqual(len(template_payload["batch_example"]), 1)

        imported = self.client.post(
            "/api/v1/admin/evaluation/dataset/import",
            json={
                "mode": "upsert",
                "samples": [
                    {
                        "id": "sample_gate_1",
                        "query": "报销审批时限是什么？",
                        "scene": "standard_qa",
                        "expected_titles": ["报销制度"],
                        "labels": ["golden"],
                    }
                ],
            },
            headers=self.headers,
        )
        self.assertEqual(imported.status_code, 200)
        self.assertEqual(imported.json()["created_count"], 1)

        exported = self.client.get("/api/v1/admin/evaluation/dataset/export?export_format=json", headers=self.headers)
        self.assertEqual(exported.status_code, 200)
        export_payload = exported.json()
        self.assertEqual(export_payload["sample_count"], 1)
        self.assertIn("sample_gate_1", export_payload["jsonl"])

        gate = self.client.get("/api/v1/admin/evaluation/release-gate?allow_review=true", headers=self.headers)
        self.assertEqual(gate.status_code, 200)
        gate_payload = gate.json()
        self.assertIn("checks", gate_payload)
        self.assertIn("release_readiness", gate_payload)

    def test_admin_can_manage_document_lifecycle(self) -> None:
        original = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "网关方案",
                "content": "旧版网关方案说明。",
                "source_connector": "feishu",
                "source_document_id": "doc_gateway_plan",
                "source_document_version": "v1",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        replacement = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "网关方案",
                "content": "新版网关方案说明。",
                "source_connector": "feishu",
                "source_document_id": "doc_gateway_plan_v2",
                "source_document_version": "v2",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(original.status_code, 200)
        self.assertEqual(replacement.status_code, 200)
        old_doc_id = original.json()["id"]
        new_doc_id = replacement.json()["id"]

        replaced = self.client.post(
            f"/api/v1/admin/documents/{old_doc_id}/replace",
            json={"replaced_by_doc_id": new_doc_id, "reason": "新版方案已上线"},
            headers=self.headers,
        )
        self.assertEqual(replaced.status_code, 200)
        replaced_payload = replaced.json()
        self.assertEqual(replaced_payload["lifecycle_status"], "deprecated")
        self.assertEqual(replaced_payload["replaced_by_doc_id"], new_doc_id)

        visible_docs = self.client.get("/api/v1/docs", headers=self.headers)
        self.assertEqual(visible_docs.status_code, 200)
        visible_ids = {item["id"] for item in visible_docs.json()}
        self.assertIn(new_doc_id, visible_ids)
        self.assertNotIn(old_doc_id, visible_ids)

        restored = self.client.post(
            f"/api/v1/admin/documents/{old_doc_id}/restore",
            json={"reason": "历史方案恢复查看"},
            headers=self.headers,
        )
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()["lifecycle_status"], "active")

        deprecated = self.client.post(
            f"/api/v1/admin/documents/{new_doc_id}/deprecate",
            json={"reason": "新版方案待修订"},
            headers=self.headers,
        )
        self.assertEqual(deprecated.status_code, 200)
        self.assertEqual(deprecated.json()["lifecycle_status"], "deprecated")

    def test_admin_can_list_stale_documents(self) -> None:
        created = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "飞书制度",
                "content": "飞书制度内容。",
                "source_connector": "feishu",
                "source_document_id": "doc_policy_1",
                "source_document_version": "v1",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(created.status_code, 200)
        doc_id = created.json()["id"]

        from app.api.deps import get_repository

        repository = get_repository()
        document = repository.get_document(doc_id)
        assert document is not None
        document.source_last_seen_at = datetime.utcnow() - timedelta(days=45)
        repository.update_document(document)

        stale = self.client.get("/api/v1/admin/documents/stale?threshold_days=30", headers=self.headers)
        self.assertEqual(stale.status_code, 200)
        stale_payload = stale.json()
        self.assertEqual(stale_payload["threshold_days"], 30)
        stale_ids = {item["id"] for item in stale_payload["documents"]}
        self.assertIn(doc_id, stale_ids)


if __name__ == "__main__":
    unittest.main()
