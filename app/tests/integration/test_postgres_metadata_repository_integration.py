import os
import unittest

import psycopg
from fastapi.testclient import TestClient


POSTGRES_DSN = "postgresql://secure_rag:secure_rag@127.0.0.1:54329/secure_rag"


class PostgresMetadataRepositoryIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._reset_metadata_tables()

        os.environ["APP_REPOSITORY_BACKEND"] = "postgres"
        os.environ["APP_POSTGRES_DSN"] = POSTGRES_DSN
        os.environ["APP_POSTGRES_AUTO_INIT_SCHEMA"] = "true"
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "30"
        os.environ["APP_ELASTICSEARCH_MODE"] = "local-fallback"
        os.environ["APP_PGVECTOR_MODE"] = "local-fallback"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings

        settings.repository_backend = "postgres"
        settings.postgres_dsn = POSTGRES_DSN
        settings.postgres_auto_init_schema = True
        settings.redis_mode = "local-fallback"
        settings.rate_limit_max_requests = 30
        settings.elasticsearch_mode = "local-fallback"
        settings.pgvector_mode = "local-fallback"
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

        cls.client = TestClient(app)
        cls.headers = {
            "X-User-Id": "u1",
            "X-Tenant-Id": "t1",
            "X-Department-Id": "engineering",
            "X-Role": "employee",
            "X-Clearance-Level": "2",
        }
        cls.get_repository = staticmethod(get_repository)

    @classmethod
    def _reset_metadata_tables(cls) -> None:
        with psycopg.connect(POSTGRES_DSN) as connection:
            connection.execute(
                """
                DROP TABLE IF EXISTS audit_logs;
                DROP TABLE IF EXISTS chat_messages;
                DROP TABLE IF EXISTS chat_sessions;
                DROP TABLE IF EXISTS document_chunks;
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS prompt_templates;
                DROP TABLE IF EXISTS policies;
                """
            )
            connection.commit()

    def test_postgres_repository_persists_documents_sessions_and_audit_logs(self) -> None:
        upload = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "采购流程",
                "content": "采购流程说明。\n\n审批时限为2个工作日。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(upload.status_code, 200)
        document = upload.json()

        query = self.client.post(
            "/api/v1/chat/query",
            json={"query": "采购审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(query.status_code, 200)
        body = query.json()
        self.assertGreaterEqual(body["retrieved_chunks"], 1)

        repository = self.get_repository()
        stored_document = repository.get_document(document["id"])
        self.assertIsNotNone(stored_document)
        self.assertEqual(stored_document.title, "采购流程")

        sessions = repository.list_sessions("t1", "u1")
        self.assertEqual(len(sessions), 1)

        messages = repository.list_messages(body["session_id"])
        self.assertEqual(len(messages), 2)

        audit_logs = repository.list_audit_logs()
        self.assertEqual(len(audit_logs), 1)
        self.assertEqual(audit_logs[0].tenant_id, "t1")


if __name__ == "__main__":
    unittest.main()
