import unittest
from pathlib import Path
from typing import Optional

from fastapi.testclient import TestClient

from app.infrastructure.external_sources.base import ExternalSourceItem, ExternalSourcePage


class _FakeFeishuClient:
    provider = "feishu"

    def __init__(self) -> None:
        self.listed_sources: list[str] = []

    def parse_source(self, source: str):
        from app.infrastructure.external_sources.feishu import FeishuSourceReference

        token = source.rstrip("/").split("/")[-1]
        return FeishuSourceReference(source_kind="wiki", token=token)

    def fetch_document(self, source: str):
        from app.infrastructure.external_sources.feishu import FeishuDocumentContent

        token = source.rstrip("/").split("/")[-1]
        return FeishuDocumentContent(
            title=f"飞书文档 {token}",
            content=f"# 飞书文档 {token}\n\n审批时限为3个工作日。",
            source_type="markdown",
            source_uri=source,
            external_document_id=token,
        )

    def health_check(self) -> dict:
        return {"backend": "feishu", "execute_enabled": True, "reachable": True}

    def list_sources(
        self,
        cursor: Optional[str] = None,
        limit: int = 20,
        source_root: Optional[str] = None,
        space_id: Optional[str] = None,
        parent_node_token: Optional[str] = None,
    ) -> ExternalSourcePage:
        start = int(cursor or "0")
        selected = self.listed_sources[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(self.listed_sources) else None
        items = [
            ExternalSourceItem(
                source=source,
                source_kind="wiki",
                external_document_id=source.rstrip("/").split("/")[-1],
                title=f"Listed {index}",
                space_id=space_id or "space_1",
                node_token=source.rstrip("/").split("/")[-1],
                parent_node_token=parent_node_token or "root_node",
                obj_type="docx",
            )
            for index, source in enumerate(selected, start=start + 1)
        ]
        return ExternalSourcePage(items=items, next_cursor=next_cursor)


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
        from app.infrastructure.cache.redis_client import RedisClient

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.openai_api_key = None
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
        self.get_feishu_source_sync_service = get_feishu_source_sync_service
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

    def test_admin_can_batch_sync_feishu_sources(self) -> None:
        self.get_feishu_source_sync_service().feishu_client = _FakeFeishuClient()
        response = self.client.post(
            "/api/v1/admin/sources/feishu/sync",
            json={
                "items": [
                    {
                        "source": "https://example.feishu.cn/wiki/wiki_token_1",
                        "department_scope": ["engineering"],
                        "async_mode": True,
                    },
                    {
                        "source": "https://example.feishu.cn/wiki/wiki_token_2",
                        "department_scope": ["engineering"],
                        "async_mode": True,
                    },
                ],
                "continue_on_error": True,
            },
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["listed_count"], 2)
        self.assertEqual(payload["succeeded"], 2)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(len(payload["items"]), 2)

    def test_admin_can_batch_sync_feishu_sources_via_connector_listing(self) -> None:
        fake_client = _FakeFeishuClient()
        fake_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
            "https://example.feishu.cn/wiki/wiki_token_3",
        ]
        self.get_feishu_source_sync_service().feishu_client = fake_client

        response = self.client.post(
            "/api/v1/admin/sources/feishu/sync",
            json={
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "cursor": "0",
                "limit": 2,
                "default_department_scope": ["engineering"],
                "default_async_mode": True,
            },
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["listed_count"], 2)
        self.assertEqual(payload["next_cursor"], "2")
        self.assertEqual(payload["succeeded"], 2)

    def test_admin_can_list_feishu_sources(self) -> None:
        fake_client = _FakeFeishuClient()
        fake_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
        ]
        self.get_feishu_source_sync_service().feishu_client = fake_client

        response = self.client.post(
            "/api/v1/admin/sources/feishu/list",
            json={
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "cursor": "0",
                "limit": 1,
            },
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["listed_count"], 1)
        self.assertEqual(payload["next_cursor"], "1")
        self.assertEqual(payload["items"][0]["source_kind"], "wiki")

    def test_admin_can_create_and_list_feishu_sync_jobs(self) -> None:
        create_response = self.client.post(
            "/api/v1/admin/sources/feishu/jobs",
            json={
                "name": "finance wiki",
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "limit": 2,
                "default_department_scope": ["engineering"],
                "default_async_mode": True,
            },
            headers=self.headers,
        )
        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        self.assertEqual(created["name"], "finance wiki")

        list_response = self.client.get("/api/v1/admin/sources/feishu/jobs", headers=self.headers)
        self.assertEqual(list_response.status_code, 200)
        payload = list_response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["job_id"], created["job_id"])
        self.assertEqual(payload[0]["source_root"], "https://example.feishu.cn/wiki/root_node")

    def test_admin_can_run_feishu_sync_job_and_advance_cursor(self) -> None:
        fake_client = _FakeFeishuClient()
        fake_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
            "https://example.feishu.cn/wiki/wiki_token_3",
        ]
        self.get_feishu_source_sync_service().feishu_client = fake_client

        create_response = self.client.post(
            "/api/v1/admin/sources/feishu/jobs",
            json={
                "name": "finance wiki",
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "limit": 2,
                "default_department_scope": ["engineering"],
                "default_async_mode": True,
            },
            headers=self.headers,
        )
        job_id = create_response.json()["job_id"]

        run_response = self.client.post(f"/api/v1/admin/sources/feishu/jobs/{job_id}/run", headers=self.headers)
        self.assertEqual(run_response.status_code, 200)
        run_payload = run_response.json()
        self.assertEqual(run_payload["listed_count"], 2)
        self.assertEqual(run_payload["next_cursor"], "2")
        self.assertIsNotNone(run_payload["run_id"])

        jobs_response = self.client.get("/api/v1/admin/sources/feishu/jobs", headers=self.headers)
        jobs_payload = jobs_response.json()
        self.assertEqual(jobs_payload[0]["cursor"], "2")
        self.assertEqual(jobs_payload[0]["last_run_status"], "success")
        self.assertEqual(jobs_payload[0]["status"], "idle")
        self.assertEqual(jobs_payload[0]["run_count"], 1)

    def test_admin_can_run_enabled_feishu_sync_jobs(self) -> None:
        fake_client = _FakeFeishuClient()
        fake_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
        ]
        self.get_feishu_source_sync_service().feishu_client = fake_client

        self.client.post(
            "/api/v1/admin/sources/feishu/jobs",
            json={
                "name": "enabled job",
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "limit": 1,
                "default_department_scope": ["engineering"],
                "default_async_mode": True,
                "enabled": True,
            },
            headers=self.headers,
        )
        self.client.post(
            "/api/v1/admin/sources/feishu/jobs",
            json={
                "name": "disabled job",
                "source_root": "https://example.feishu.cn/wiki/root_node",
                "limit": 1,
                "default_department_scope": ["engineering"],
                "default_async_mode": True,
                "enabled": False,
            },
            headers=self.headers,
        )

        response = self.client.post("/api/v1/admin/sources/feishu/jobs/run-enabled", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_jobs"], 2)
        self.assertEqual(payload["succeeded_jobs"], 1)
        self.assertEqual(payload["failed_jobs"], 0)
        self.assertEqual(payload["skipped_jobs"], 1)

    def test_admin_can_view_feishu_sync_runs(self) -> None:
        self.get_feishu_source_sync_service().feishu_client = _FakeFeishuClient()
        self.client.post(
            "/api/v1/admin/sources/feishu/sync",
            json={
                "items": [
                    {
                        "source": "https://example.feishu.cn/wiki/wiki_token_1",
                        "department_scope": ["engineering"],
                        "async_mode": True,
                    }
                ]
            },
            headers=self.headers,
        )
        response = self.client.get("/api/v1/admin/sources/feishu/runs", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["provider"], "feishu")
        self.assertEqual(payload[0]["status"], "success")

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
        self.assertGreater(payload["intent_confidence"], 0)
        self.assertIsInstance(payload["intent_reasons"], list)
        self.assertIn("understanding_source", payload)
        self.assertIn("rule_intent", payload)
        self.assertIn("rule_rewritten_query", payload)
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
        self.assertEqual(log["conversation_json"]["intent"], "standard_qa")
        self.assertGreater(log["conversation_json"]["intent_confidence"], 0)
        self.assertIsInstance(log["conversation_json"]["intent_reasons"], list)
        self.assertIn("understanding_source", log["conversation_json"])
        self.assertIn("rule_intent", log["conversation_json"])
        self.assertIn("rule_rewritten_query", log["conversation_json"])


if __name__ == "__main__":
    unittest.main()
