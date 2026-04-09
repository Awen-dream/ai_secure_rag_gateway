import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class ChatConversationFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_conversation_flow.db"
        Path(self.db_path).unlink(missing_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings
        from app.infrastructure.cache.redis_client import RedisClient

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.openai_api_key = None
        RedisClient.reset_local_state()

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
            get_query_planning_service,
            get_recall_planning_service,
            get_query_understanding_service,
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
            get_query_understanding_service,
            get_query_planning_service,
            get_recall_planning_service,
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
            "X-Role": "employee",
            "X-Clearance-Level": "2",
        }

        self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "报销制度",
                "content": "报销制度说明。\n\n审批时限为3个工作日。\n\n超标费用需要审批。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "接口规范",
                "content": "接口规范说明。\n\n认证方式为 OAuth2。\n\n错误码见附录。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )

    def test_follow_up_query_returns_rewritten_query(self) -> None:
        first = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销制度是什么？"},
            headers=self.headers,
        )
        self.assertEqual(first.status_code, 200)
        session_id = first.json()["session_id"]

        follow_up = self.client.post(
            "/api/v1/chat/query",
            json={"query": "审批时限呢？", "session_id": session_id},
            headers=self.headers,
        )
        self.assertEqual(follow_up.status_code, 200)
        payload = follow_up.json()
        self.assertIn("报销制度是什么", payload["rewritten_query"])
        self.assertFalse(payload["topic_switched"])

    def test_topic_switch_is_detected(self) -> None:
        first = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销制度是什么？"},
            headers=self.headers,
        )
        session_id = first.json()["session_id"]

        switched = self.client.post(
            "/api/v1/chat/query",
            json={"query": "这个接口规范在哪里？", "session_id": session_id},
            headers=self.headers,
        )
        self.assertEqual(switched.status_code, 200)
        payload = switched.json()
        self.assertEqual(payload["rewritten_query"], "这个接口规范在哪里？")
        self.assertTrue(payload["topic_switched"])


if __name__ == "__main__":
    unittest.main()
