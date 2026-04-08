import shutil
import unittest
from pathlib import Path
from typing import Optional

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.auth.models import UserContext
from app.domain.documents.services import DocumentService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.sources.schemas import FeishuBatchSyncRequest, FeishuImportRequest
from app.domain.sources.services import FeishuSourceSyncService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.external_sources.base import ExternalSourceItem, ExternalSourcePage
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore


class _FakeFeishuClient:
    provider = "feishu"

    def __init__(self) -> None:
        self.content = "# 飞书报销制度\n\n审批时限为3个工作日。"
        self.fail_sources: set[str] = set()
        self.listed_sources: list[str] = []

    def parse_source(self, source: str):
        from app.infrastructure.external_sources.feishu import FeishuSourceReference

        return FeishuSourceReference(source_kind="wiki", token="wiki_token")

    def fetch_document(self, source: str):
        if source in self.fail_sources:
            raise RuntimeError("simulated fetch failure")
        from app.infrastructure.external_sources.feishu import FeishuDocumentContent

        return FeishuDocumentContent(
            title="飞书报销制度",
            content=self.content,
            source_type="markdown",
            source_uri=source,
            external_document_id="wiki_token",
        )

    def health_check(self) -> dict:
        return {"backend": "feishu", "execute_enabled": True, "reachable": True}

    def list_sources(self, cursor: Optional[str] = None, limit: int = 20) -> ExternalSourcePage:
        start = int(cursor or "0")
        selected = self.listed_sources[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(self.listed_sources) else None
        items = [
            ExternalSourceItem(
                source=source,
                source_kind="wiki",
                external_document_id=source.rstrip("/").split("/")[-1],
                title=f"Listed {index}",
            )
            for index, source in enumerate(selected, start=start + 1)
        ]
        return ExternalSourcePage(items=items, next_cursor=next_cursor)


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
        self.feishu_client = _FakeFeishuClient()
        self.service = FeishuSourceSyncService(
            feishu_client=self.feishu_client,
            repository=repository,
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
        self.assertEqual(document.source_connector, "feishu")
        self.assertEqual(document.source_document_id, "wiki_token")
        self.assertEqual(response.sync_action, "imported_new")

    def test_repeated_sync_of_same_external_document_reuses_current_version(self) -> None:
        first = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=False,
            ),
            self.user,
        )
        second = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=False,
            ),
            self.user,
        )

        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(second.sync_action, "reused_current")
        history = self.repository.list_documents_by_source_ref("t1", "feishu", "wiki_token")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].version, 1)

    def test_changed_external_document_content_creates_new_version(self) -> None:
        first = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=False,
            ),
            self.user,
        )

        self.feishu_client.content = "# 飞书报销制度\n\n审批时限为5个工作日。"
        second = self.service.import_source(
            FeishuImportRequest(
                source="https://example.feishu.cn/wiki/wiki_token",
                department_scope=["finance"],
                async_mode=False,
            ),
            self.user,
        )

        self.assertNotEqual(first.document_id, second.document_id)
        self.assertEqual(second.sync_action, "created_new_version")
        history = self.repository.list_documents_by_source_ref("t1", "feishu", "wiki_token")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].version, 2)
        self.assertTrue(history[0].current)
        self.assertFalse(history[1].current)

    def test_batch_sync_returns_summary_counts(self) -> None:
        response = self.service.sync_sources(
            FeishuBatchSyncRequest(
                items=[
                    FeishuImportRequest(
                        source="https://example.feishu.cn/wiki/wiki_token",
                        department_scope=["finance"],
                        async_mode=False,
                    ),
                    FeishuImportRequest(
                        source="https://example.feishu.cn/wiki/wiki_token",
                        department_scope=["finance"],
                        async_mode=False,
                    ),
                ]
            ),
            self.user,
        )

        self.assertEqual(response.total, 2)
        self.assertEqual(response.listed_count, 2)
        self.assertEqual(response.succeeded, 2)
        self.assertEqual(response.failed, 0)
        self.assertEqual(response.imported_new, 1)
        self.assertEqual(response.reused_current, 1)
        self.assertEqual(response.created_new_version, 0)
        runs = self.repository.list_source_sync_runs("t1", "feishu")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "success")
        self.assertEqual(runs[0].succeeded, 2)
        self.assertEqual(len(runs[0].result_items_json), 2)

    def test_batch_sync_can_continue_after_item_failure(self) -> None:
        self.feishu_client.fail_sources.add("https://example.feishu.cn/wiki/bad_token")

        response = self.service.sync_sources(
            FeishuBatchSyncRequest(
                items=[
                    FeishuImportRequest(
                        source="https://example.feishu.cn/wiki/bad_token",
                        department_scope=["finance"],
                        async_mode=False,
                    ),
                    FeishuImportRequest(
                        source="https://example.feishu.cn/wiki/wiki_token",
                        department_scope=["finance"],
                        async_mode=False,
                    ),
                ],
                continue_on_error=True,
            ),
            self.user,
        )

        self.assertEqual(response.total, 2)
        self.assertEqual(response.listed_count, 2)
        self.assertEqual(response.succeeded, 1)
        self.assertEqual(response.failed, 1)
        self.assertEqual(response.items[0].sync_action, "failed")
        self.assertTrue(response.items[1].success)
        runs = self.repository.list_source_sync_runs("t1", "feishu")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "partial_success")

    def test_batch_sync_can_list_sources_by_cursor_when_items_are_not_provided(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
            "https://example.feishu.cn/wiki/wiki_token_3",
        ]

        response = self.service.sync_sources(
            FeishuBatchSyncRequest(
                cursor="0",
                limit=2,
                default_department_scope=["finance"],
                default_async_mode=False,
            ),
            self.user,
        )

        self.assertEqual(response.total, 2)
        self.assertEqual(response.listed_count, 2)
        self.assertEqual(response.next_cursor, "2")
        self.assertEqual(response.succeeded, 2)
        self.assertEqual(response.items[0].source, "https://example.feishu.cn/wiki/wiki_token_1")
        self.assertEqual(response.items[1].source, "https://example.feishu.cn/wiki/wiki_token_2")


if __name__ == "__main__":
    unittest.main()
