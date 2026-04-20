from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.auth.models import UserContext
from app.domain.sources.schemas import FeishuBatchSyncRequest, FeishuSyncJobUpsertRequest
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.frameworks.langgraph_ingestion import LangGraphDocumentIngestionEngine
from app.infrastructure.frameworks.langgraph_source_sync import LangGraphSourceSyncWorkflow
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.documents.services import DocumentService


class _FakeCompiledGraph:
    def __init__(self, node_callable):
        self.node_callable = node_callable

    def invoke(self, state):
        return self.node_callable(state)


class _FakeStateGraph:
    def __init__(self, _state_type):
        self._node_callable = None

    def add_node(self, _name, node_callable):
        self._node_callable = node_callable

    def set_entry_point(self, _name):
        return None

    def add_edge(self, _start, _end):
        return None

    def compile(self):
        return _FakeCompiledGraph(self._node_callable)


class _FakeFeishuClient:
    provider = "feishu"

    def parse_source(self, source: str):
        return SimpleNamespace(source_kind="wiki", token=source.rstrip("/").split("/")[-1])

    def fetch_document(self, source: str):
        token = source.rstrip("/").split("/")[-1]
        return SimpleNamespace(
            title=f"飞书文档 {token}",
            content="# 标题\n\n正文内容",
            source_type="markdown",
            source_uri=source,
            connector="feishu",
            external_document_id=token,
            external_version=None,
        )

    def list_sources(self, **_kwargs):
        return SimpleNamespace(items=[], next_cursor=None)


class LangGraphWorkflowTest(unittest.TestCase):
    def test_langgraph_ingestion_engine_can_build_workflow(self) -> None:
        engine = LangGraphDocumentIngestionEngine()
        with patch(
            "app.infrastructure.frameworks.langgraph_ingestion.import_module",
            return_value=SimpleNamespace(StateGraph=_FakeStateGraph, END="END"),
        ):
            workflow = engine._build_workflow(
                repository=SimpleNamespace(),
                indexing_service=SimpleNamespace(),
                source_store=SimpleNamespace(),
                retrieval_cache=None,
            )

        self.assertIsNotNone(workflow)
        self.assertTrue(hasattr(workflow, "invoke"))

    def test_langgraph_source_sync_workflow_wraps_native_run(self) -> None:
        workflow_builder = LangGraphSourceSyncWorkflow()
        with patch(
            "app.infrastructure.frameworks.langgraph_source_sync.import_module",
            return_value=SimpleNamespace(StateGraph=_FakeStateGraph, END="END"),
        ):
            workflow = workflow_builder.build_run_sync_job_workflow(
                lambda job_id, user: {"job_id": job_id, "tenant_id": user.tenant_id}
            )

        self.assertIsNotNone(workflow)
        state = workflow.invoke({"job_id": "job_1", "user": SimpleNamespace(tenant_id="t1")})
        self.assertEqual(state["summary"]["job_id"], "job_1")

    def test_feishu_source_sync_service_uses_langgraph_workflow_when_available(self) -> None:
        db_path = "/tmp/secure_rag_gateway_langgraph_sync.db"
        staging_dir = "/tmp/secure_rag_gateway_langgraph_sync_staging"
        Path(db_path).unlink(missing_ok=True)
        Path(staging_dir).mkdir(parents=True, exist_ok=True)
        try:
            repository = SQLiteRepository(db_path)
            source_store = LocalDocumentSourceStore(staging_dir)
            indexing_service = RetrievalIndexingService(
                keyword_backend=ElasticsearchSearch(index_name="test_chunks", mode="local-fallback"),
                vector_backend=PGVectorStore(table_name="test_vectors", embedding_dimension=64, mode="local-fallback"),
            )
            orchestrator = DocumentIngestionOrchestrator(repository, indexing_service, source_store)
            document_service = DocumentService(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                ingestion_orchestrator=orchestrator,
            )
            service = FeishuSourceSyncService(
                feishu_client=_FakeFeishuClient(),
                repository=repository,
                document_service=document_service,
                task_queue=DocumentIngestionTaskQueue(RedisClient(), "queue:test_langgraph_sync"),
                ingestion_orchestrator=orchestrator,
                source_sync_workflow=LangGraphSourceSyncWorkflow(),
            )
            user = UserContext(
                user_id="u1",
                tenant_id="t1",
                department_id="finance",
                role="admin",
                clearance_level=3,
            )
            service.upsert_sync_job(
                FeishuSyncJobUpsertRequest(
                    name="job",
                    source_root="https://example.feishu.cn/wiki/root",
                    enabled=True,
                ),
                user,
            )
            job_id = service.list_sync_jobs(user)[0].job_id

            with patch(
                "app.infrastructure.frameworks.langgraph_source_sync.import_module",
                return_value=SimpleNamespace(StateGraph=_FakeStateGraph, END="END"),
            ):
                response = service.run_sync_job(job_id, user)

            self.assertEqual(response.total, 0)
        finally:
            Path(db_path).unlink(missing_ok=True)
            for child in Path(staging_dir).glob("*"):
                child.unlink(missing_ok=True)
            Path(staging_dir).rmdir()
