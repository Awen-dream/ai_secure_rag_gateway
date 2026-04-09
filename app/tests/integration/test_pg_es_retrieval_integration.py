import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class PgEsRetrievalIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = "/tmp/secure_rag_gateway_pg_es_integration.db"
        Path(cls.db_path).unlink(missing_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = cls.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "30"
        os.environ["APP_ELASTICSEARCH_MODE"] = "remote"
        os.environ["APP_ELASTICSEARCH_ENDPOINT"] = "http://127.0.0.1:9200"
        os.environ["APP_ELASTICSEARCH_AUTO_INIT_INDEX"] = "true"
        os.environ["APP_PGVECTOR_MODE"] = "postgres"
        os.environ["APP_PGVECTOR_DSN"] = "postgresql://secure_rag:secure_rag@127.0.0.1:54329/secure_rag"
        os.environ["APP_PGVECTOR_AUTO_INIT_SCHEMA"] = "true"
        os.environ["APP_EMBEDDING_DIMENSION"] = "256"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings

        settings.repository_backend = "sqlite"
        settings.sqlite_path = cls.db_path
        settings.redis_mode = "local-fallback"
        settings.rate_limit_max_requests = 30
        settings.elasticsearch_mode = "remote"
        settings.elasticsearch_endpoint = "http://127.0.0.1:9200"
        settings.elasticsearch_auto_init_index = True
        settings.pgvector_mode = "postgres"
        settings.pgvector_dsn = "postgresql://secure_rag:secure_rag@127.0.0.1:54329/secure_rag"
        settings.pgvector_auto_init_schema = True
        settings.embedding_dimension = 256
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
            get_prompt_template_service,
            get_query_planning_service,
            get_query_understanding_service,
            get_rate_limit_service,
            get_recall_planning_service,
            get_redis_client,
            get_repository,
            get_retrieval_rerank_service,
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
            get_prompt_template_service,
            get_policy_engine,
            get_output_guard,
            get_audit_service,
            get_openai_client,
            get_query_understanding_service,
            get_query_planning_service,
            get_recall_planning_service,
            get_retrieval_rerank_service,
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
            "X-Role": "admin",
            "X-Clearance-Level": "3",
        }

    def test_backend_health_is_reachable(self) -> None:
        es_response = self.client.get("/api/v1/admin/retrieval/backends/elasticsearch/health", headers=self.headers)
        self.assertEqual(es_response.status_code, 200)
        self.assertTrue(es_response.json()["execute_enabled"])
        self.assertTrue(es_response.json()["reachable"])

        pg_response = self.client.get("/api/v1/admin/retrieval/backends/pgvector/health", headers=self.headers)
        self.assertEqual(pg_response.status_code, 200)
        self.assertTrue(pg_response.json()["execute_enabled"])
        self.assertTrue(pg_response.json()["reachable"])

    def test_index_init_endpoints_execute(self) -> None:
        es_response = self.client.post("/api/v1/admin/retrieval/backends/elasticsearch/init-index", headers=self.headers)
        self.assertEqual(es_response.status_code, 200)
        self.assertTrue(es_response.json()["executed"])

        pg_response = self.client.post("/api/v1/admin/retrieval/backends/pgvector/init-schema", headers=self.headers)
        self.assertEqual(pg_response.status_code, 200)
        self.assertTrue(pg_response.json()["executed"])

    def test_upload_indexes_into_pg_and_es_and_query_hits(self) -> None:
        upload = self.client.post(
            "/api/v1/docs/upload",
            json={
                "title": "报销制度",
                "content": "报销制度说明。\n\n审批时限为3个工作日。\n\n超标费用需要部门负责人审批。",
                "department_scope": ["engineering"],
                "security_level": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(upload.status_code, 200)

        explain = self.client.post(
            "/api/v1/admin/retrieval/explain",
            json={"query": "报销审批时限是什么？", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(explain.status_code, 200)
        payload = explain.json()
        self.assertGreaterEqual(len(payload["results"]), 1)
        self.assertIn("elasticsearch", payload["results"][0]["retrieval_sources"])
        self.assertIn("pgvector", payload["results"][0]["retrieval_sources"])

        query = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(query.status_code, 200)
        body = query.json()
        self.assertGreaterEqual(body["retrieved_chunks"], 1)
        self.assertIn("审批时限为3个工作日", body["answer"])
