import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class ChatRateLimitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_rate_limit.db"
        Path(self.db_path).unlink(missing_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["APP_RATE_LIMIT_WINDOW_SECONDS"] = "60"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "1"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.rate_limit_window_seconds = 60
        settings.rate_limit_max_requests = 1
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

        self.client = TestClient(app)
        self.headers = {
            "X-User-Id": "u-rate-limit",
            "X-Tenant-Id": "t1",
            "X-Department-Id": "engineering",
            "X-Role": "employee",
            "X-Clearance-Level": "2",
        }
        self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "报销制度",
                "content": "审批时限为3个工作日。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )

    def test_chat_query_returns_429_after_limit(self) -> None:
        first = self.client.post("/api/v1/chat/query", json={"query": "报销审批时限是什么？"}, headers=self.headers)
        second = self.client.post("/api/v1/chat/query", json={"query": "再问一次"}, headers=self.headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"], "Rate limit exceeded.")


if __name__ == "__main__":
    unittest.main()
