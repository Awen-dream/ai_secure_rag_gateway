import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.domain.auth.models import UserContext


class RedisIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = "/tmp/secure_rag_gateway_redis_integration.db"
        Path(cls.db_path).unlink(missing_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = cls.db_path
        os.environ["APP_REDIS_MODE"] = "redis"
        os.environ["APP_REDIS_URL"] = "redis://127.0.0.1:63799/0"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "30"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings

        settings.repository_backend = "sqlite"
        settings.sqlite_path = cls.db_path
        settings.redis_mode = "redis"
        settings.redis_url = "redis://127.0.0.1:63799/0"
        settings.rate_limit_max_requests = 30
        settings.openai_api_key = None

        from app.api.deps import (
            get_audit_service,
            get_chat_service,
            get_document_ingestion_worker,
            get_document_service,
            get_document_ingestion_orchestrator,
            get_document_source_store,
            get_document_task_queue,
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
            get_document_task_queue,
            get_keyword_backend,
            get_vector_backend,
            get_document_source_store,
            get_indexing_service,
            get_document_ingestion_orchestrator,
            get_document_ingestion_worker,
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

        cls.client = TestClient(app)
        cls.headers = {
            "X-User-Id": "u1",
            "X-Tenant-Id": "t1",
            "X-Department-Id": "engineering",
            "X-Role": "employee",
            "X-Clearance-Level": "2",
        }
        cls.get_redis_client = staticmethod(get_redis_client)
        cls.get_retrieval_cache = staticmethod(get_retrieval_cache)
        cls.get_session_cache = staticmethod(get_session_cache)
        cls.get_document_task_queue = staticmethod(get_document_task_queue)

    def test_admin_cache_health_and_runtime_cache_keys(self) -> None:
        health = self.client.get(
            "/api/v1/admin/cache/health",
            headers={
                **self.headers,
                "X-Role": "admin",
            },
        )
        self.assertEqual(health.status_code, 200)
        payload = health.json()
        self.assertTrue(payload["execute_enabled"])
        self.assertTrue(payload["reachable"])

        upload = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "报销制度",
                "content": "报销制度说明。\n\n审批时限为3个工作日。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(upload.status_code, 200)

        query = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(query.status_code, 200)
        body = query.json()

        user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="engineering",
            role="employee",
            clearance_level=2,
        )
        session_summary = self.get_session_cache().get_summary(body["session_id"])
        cached_results = self.get_retrieval_cache().get_results(user, "报销审批时限是什么？", 5)

        self.assertIsNotNone(session_summary)
        self.assertIn("assistant:", session_summary)
        self.assertIsNotNone(cached_results)
        self.assertGreaterEqual(len(cached_results), 1)
        self.assertTrue(self.get_redis_client().ping())

    def test_document_ingestion_queue_uses_real_redis(self) -> None:
        health = self.client.get(
            "/api/v1/admin/queue/document-ingestion/health",
            headers={
                **self.headers,
                "X-Role": "admin",
            },
        )
        self.assertEqual(health.status_code, 200)
        payload = health.json()
        self.assertTrue(payload["execute_enabled"])
        self.assertTrue(payload["reachable"])

        queue = self.get_document_task_queue()
        receipt = queue.enqueue_document("doc_queue_1")
        self.assertEqual(receipt["status"], "queued")
        self.assertGreaterEqual(queue.queue_depth(), 1)

        task = queue.dequeue_document(timeout_seconds=1)
        self.assertIsNotNone(task)
        self.assertEqual(task["doc_id"], "doc_queue_1")


if __name__ == "__main__":
    unittest.main()
