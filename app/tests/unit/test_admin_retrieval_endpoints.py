import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class AdminRetrievalEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_admin_retrieval.db"
        Path(self.db_path).unlink(missing_ok=True)

        import os

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["OPENAI_API_KEY"] = ""
        from app.core.config import settings

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.openai_api_key = None

        from app.api.deps import (
            get_audit_service,
            get_chat_service,
            get_document_ingestion_worker,
            get_document_service,
            get_document_ingestion_orchestrator,
            get_document_source_store,
            get_document_task_queue,
            get_embedding_client,
            get_feishu_client,
            get_feishu_source_sync_service,
            get_indexing_service,
            get_keyword_backend,
            get_openai_client,
            get_output_guard,
            get_policy_engine,
            get_prompt_service,
            get_rate_limit_service,
            get_redis_client,
            get_repository,
            get_retrieval_reranker,
            get_retrieval_cache,
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
            get_retrieval_reranker,
            get_prompt_service,
            get_policy_engine,
            get_output_guard,
            get_audit_service,
            get_openai_client,
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
            "X-Clearance-Level": "3",
        }

    def test_admin_can_view_backend_info(self) -> None:
        response = self.client.get("/api/v1/admin/retrieval/backends", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["backend"], "elasticsearch")
        self.assertEqual(payload[1]["backend"], "pgvector")

    def test_admin_can_view_cache_health(self) -> None:
        response = self.client.get("/api/v1/admin/cache/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "redis")
        self.assertFalse(payload["execute_enabled"])
        self.assertTrue(payload["reachable"])

    def test_admin_can_view_document_ingestion_queue_health(self) -> None:
        response = self.client.get("/api/v1/admin/queue/document-ingestion/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "redis")
        self.assertFalse(payload["execute_enabled"])
        self.assertTrue(payload["reachable"])
        self.assertIn("queue_depth", payload)

    def test_admin_can_view_feishu_source_health(self) -> None:
        response = self.client.get("/api/v1/admin/sources/feishu/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "feishu")
        self.assertFalse(payload["execute_enabled"])

    def test_admin_can_enable_preview_and_validate_prompt_templates(self) -> None:
        created = self.client.post(
            "/api/v1/admin/prompts",
            json={
                "id": "prompt_standard_v2",
                "scene": "standard_qa",
                "version": 2,
                "name": "Standard QA V2",
                "content": "Use citations and answer conservatively.",
                "output_schema": {"sections": "结论,依据,引用来源,限制说明"},
                "enabled": False,
                "created_by": "admin",
            },
            headers=self.headers,
        )
        self.assertEqual(created.status_code, 200)
        self.assertFalse(created.json()["enabled"])

        enabled = self.client.post(
            "/api/v1/admin/prompts/prompt_standard_v2/enable",
            params={"enabled": "true"},
            headers=self.headers,
        )
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json()["enabled"])

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
        preview = self.client.post(
            "/api/v1/admin/prompts/preview",
            json={"scene": "standard_qa", "query": "报销审批时限是什么？", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(preview.status_code, 200)
        preview_payload = preview.json()
        self.assertEqual(preview_payload["template_id"], "prompt_standard_v2")
        self.assertIn("用户问题", preview_payload["input_text"])

        validation = self.client.post(
            "/api/v1/admin/prompts/validate",
            json={"scene": "standard_qa", "answer": "结论：已回答。\n依据：来自授权资料。"},
            headers=self.headers,
        )
        self.assertEqual(validation.status_code, 200)
        validation_payload = validation.json()
        self.assertFalse(validation_payload["valid"])
        self.assertIn("引用来源", validation_payload["missing_sections"])
        self.assertIn("限制说明", validation_payload["normalized_answer"])

    def test_admin_can_view_backend_plan(self) -> None:
        es_response = self.client.get(
            "/api/v1/admin/retrieval/backends/elasticsearch/plan",
            params={"query": "报销审批时限是什么", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(es_response.status_code, 200)
        es_payload = es_response.json()
        self.assertEqual(es_payload["backend"], "elasticsearch")
        self.assertIn("access_filter", es_payload["artifacts"])
        self.assertIn("mapping", es_payload["artifacts"])
        self.assertIn("search_body", es_payload["artifacts"])

        pg_response = self.client.get(
            "/api/v1/admin/retrieval/backends/pgvector/plan",
            params={"query": "报销审批时限是什么", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(pg_response.status_code, 200)
        pg_payload = pg_response.json()
        self.assertEqual(pg_payload["backend"], "pgvector")
        self.assertIn("access_filter", pg_payload["artifacts"])
        self.assertIn("ddl", pg_payload["artifacts"])
        self.assertIn("upsert_sql", pg_payload["artifacts"])
        self.assertIn("search_sql", pg_payload["artifacts"])

    def test_admin_can_view_backend_health(self) -> None:
        es_response = self.client.get("/api/v1/admin/retrieval/backends/elasticsearch/health", headers=self.headers)
        self.assertEqual(es_response.status_code, 200)
        es_payload = es_response.json()
        self.assertEqual(es_payload["backend"], "elasticsearch")
        self.assertFalse(es_payload["execute_enabled"])
        self.assertFalse(es_payload["reachable"])

        pg_response = self.client.get("/api/v1/admin/retrieval/backends/pgvector/health", headers=self.headers)
        self.assertEqual(pg_response.status_code, 200)
        pg_payload = pg_response.json()
        self.assertEqual(pg_payload["backend"], "pgvector")
        self.assertFalse(pg_payload["execute_enabled"])
        self.assertFalse(pg_payload["reachable"])

    def test_elasticsearch_init_index_is_safe_in_fallback_mode(self) -> None:
        response = self.client.post("/api/v1/admin/retrieval/backends/elasticsearch/init-index", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "elasticsearch")
        self.assertFalse(payload["executed"])

    def test_pgvector_init_schema_is_safe_in_fallback_mode(self) -> None:
        response = self.client.post("/api/v1/admin/retrieval/backends/pgvector/init-schema", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "pgvector")
        self.assertFalse(payload["executed"])

    def test_admin_can_explain_retrieval(self) -> None:
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

        response = self.client.post(
            "/api/v1/admin/retrieval/explain",
            json={"query": "报销审批时限是什么？", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "standard_qa")
        self.assertGreaterEqual(len(payload["results"]), 1)
        self.assertIn("elasticsearch", payload["results"][0]["retrieval_sources"])
        self.assertIn("pgvector", payload["results"][0]["retrieval_sources"])

    def test_admin_can_view_structured_audit_logs(self) -> None:
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
        query = self.client.post(
            "/api/v1/chat/query",
            json={"query": "采购审批时限呢？"},
            headers=self.headers,
        )
        self.assertEqual(query.status_code, 200)

        audit = self.client.get("/api/v1/admin/audit", headers=self.headers)
        self.assertEqual(audit.status_code, 200)
        payload = audit.json()
        self.assertEqual(len(payload), 1)
        log = payload[0]
        self.assertEqual(log["query"], "采购审批时限呢？")
        self.assertEqual(log["scene"], "standard_qa")
        self.assertIn("rewritten_query", log)
        self.assertIn("prompt_json", log)
        self.assertIn("risk_json", log)
        self.assertIn("conversation_json", log)
        self.assertGreaterEqual(len(log["retrieval_docs_json"]), 1)
        self.assertIn("score", log["retrieval_docs_json"][0])
        self.assertIn("template_id", log["prompt_json"])
        self.assertIn("final_action", log["risk_json"])


if __name__ == "__main__":
    unittest.main()
