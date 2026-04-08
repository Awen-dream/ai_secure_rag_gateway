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
            get_document_service,
            get_indexing_service,
            get_keyword_backend,
            get_openai_client,
            get_output_guard,
            get_policy_engine,
            get_prompt_service,
            get_rate_limit_service,
            get_redis_client,
            get_repository,
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
            get_keyword_backend,
            get_vector_backend,
            get_indexing_service,
            get_document_service,
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


if __name__ == "__main__":
    unittest.main()
