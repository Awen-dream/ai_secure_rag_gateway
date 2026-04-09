import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class ChatLLMIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_chat_llm.db"
        Path(self.db_path).unlink(missing_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "30"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_MODEL"] = "gpt-5.4-mini"

        from app.core.config import settings
        from app.infrastructure.cache.redis_client import RedisClient

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.rate_limit_max_requests = 30
        settings.openai_api_key = "test-key"
        settings.openai_model = "gpt-5.4-mini"
        RedisClient.reset_local_state()

        from app.api.deps import (
            get_audit_service,
            get_chat_service,
            get_context_builder_service,
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
            get_prompt_builder_service,
            get_generation_service,
            get_query_planning_service,
            get_recall_planning_service,
            get_retrieval_rerank_service,
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
            get_context_builder_service,
            get_audit_service,
            get_openai_client,
            get_prompt_builder_service,
            get_generation_service,
            get_query_understanding_service,
            get_query_planning_service,
            get_recall_planning_service,
            get_retrieval_rerank_service,
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

    def tearDown(self) -> None:
        os.environ["OPENAI_API_KEY"] = ""

    def test_chat_query_uses_openai_client_when_configured(self) -> None:
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

        mocked_answer = "结论：审批时限为3个工作日。\n依据：[1] 报销制度。\n引用来源：[1] 报销制度 / Section 1 / v1"
        with patch(
            "app.infrastructure.llm.openai_client.OpenAIClient.generate_response",
            return_value=mocked_answer,
        ) as generate_response:
            response = self.client.post(
                "/api/v1/chat/query",
                json={"query": "报销审批时限是什么？"},
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], mocked_answer + "\n限制说明：无。")
        self.assertEqual(payload["risk_action"], "allow")
        self.assertEqual(len(payload["citations"]), 1)
        generate_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
