import shutil
import unittest
from pathlib import Path

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.sources.schemas import FeishuImportRequest
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore


class _FakeFeishuClient:
    def parse_source(self, source: str):
        from app.infrastructure.external_sources.feishu import FeishuSourceReference

        return FeishuSourceReference(source_kind="wiki", token="wiki_token")

    def fetch_document(self, source: str):
        from app.infrastructure.external_sources.feishu import FeishuDocumentContent

        return FeishuDocumentContent(
            title="飞书报销制度",
            content="# 飞书报销制度\n\n审批时限为3个工作日。",
            source_type="markdown",
            source_uri=source,
        )

    def health_check(self) -> dict:
        return {"backend": "feishu", "execute_enabled": True, "reachable": True}


class FeishuSourceSyncServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_feishu_sync.db"
        self.staging_dir = Path("/tmp/secure_rag_gateway_feishu_staging")
        Path(self.db_path).unlink(missing_ok=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)

        repository = SQLiteRepository(self.db_path)
        source_store = LocalDocumentSourceStore(str(self.staging_dir))
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
        task_queue = DocumentIngestionTaskQueue(
            redis_client=RedisClient(),
            queue_name="queue:feishu_sync_test",
        )

        self.repository = repository
        self.orchestrator = orchestrator
        self.task_queue = task_queue
        self.service = FeishuSourceSyncService(
            feishu_client=_FakeFeishuClient(),
            document_service=document_service,
            task_queue=task_queue,
            ingestion_orchestrator=orchestrator,
        )
        self.user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="finance",
            role="admin",
            clearance_level=3,
        )

    def tearDown(self) -> None:
        Path(self.db_path).unlink(missing_ok=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    def test_import_source_can_queue_document_ingestion(self) -> None:
        response = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=True,
            ),
            self.user,
        )

        self.assertTrue(response.queued)
        self.assertEqual(self.task_queue.queue_depth(), 1)
        document = self.repository.get_document(response.document_id)
        self.assertIsNotNone(document)
        self.assertEqual(document.status.value, "pending")

    def test_import_source_can_process_synchronously(self) -> None:
        response = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=False,
            ),
            self.user,
        )

        self.assertFalse(response.queued)
        document = self.repository.get_document(response.document_id)
        self.assertIsNotNone(document)
        self.assertEqual(document.status.value, "success")
        self.assertGreaterEqual(len(self.repository.list_chunks_for_document(response.document_id)), 1)


if __name__ == "__main__":
    unittest.main()
