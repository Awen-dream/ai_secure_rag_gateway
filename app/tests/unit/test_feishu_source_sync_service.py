import shutil
import unittest
from pathlib import Path
from typing import Optional

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentStatus
from app.domain.documents.services import DocumentService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.sources.schemas import (
    FeishuBatchSyncRequest,
    FeishuImportRequest,
    FeishuListSourcesRequest,
    FeishuSyncJobUpsertRequest,
)
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

        return FeishuSourceReference(source_kind="wiki", token=source.rstrip("/").split("/")[-1])

    def fetch_document(self, source: str):
        if source in self.fail_sources:
            raise RuntimeError("simulated fetch failure")
        from app.infrastructure.external_sources.feishu import FeishuDocumentContent

        token = source.rstrip("/").split("/")[-1]
        return FeishuDocumentContent(
            title=f"飞书报销制度 {token}",
            content=self.content,
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
                source_root="https://example.feishu.cn/wiki/root_node",
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

    def test_list_sources_returns_connector_page(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
        ]

        response = self.service.list_sources(
            FeishuListSourcesRequest(
                source_root="https://example.feishu.cn/wiki/root_node",
                cursor="0",
                limit=1,
            )
        )

        self.assertEqual(response.listed_count, 1)
        self.assertEqual(response.next_cursor, "1")
        self.assertEqual(response.items[0].source_kind, "wiki")
        self.assertEqual(response.items[0].obj_type, "docx")

    def test_upsert_and_run_sync_job_persists_cursor_checkpoint(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
            "https://example.feishu.cn/wiki/wiki_token_3",
        ]

        job = self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="finance wiki",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=2,
                default_department_scope=["finance"],
                default_async_mode=False,
            ),
            self.user,
        )

        first_run = self.service.run_sync_job(job.job_id, self.user)
        self.assertEqual(first_run.listed_count, 2)
        self.assertEqual(first_run.next_cursor, "2")

        saved_job = self.repository.get_source_sync_job("t1", "feishu", job.job_id)
        self.assertIsNotNone(saved_job)
        self.assertEqual(saved_job.cursor, "2")
        self.assertEqual(saved_job.last_run_status, "success")
        self.assertIsNotNone(saved_job.last_run_id)

        second_run = self.service.run_sync_job(job.job_id, self.user)
        self.assertEqual(second_run.listed_count, 1)
        self.assertIsNone(second_run.next_cursor)

        saved_job = self.repository.get_source_sync_job("t1", "feishu", job.job_id)
        self.assertIsNotNone(saved_job)
        self.assertIsNone(saved_job.cursor)

    def test_list_sync_jobs_returns_saved_jobs(self) -> None:
        created = self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="finance wiki",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=5,
            ),
            self.user,
        )

        jobs = self.service.list_sync_jobs(self.user)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, created.job_id)
        self.assertEqual(jobs[0].name, "finance wiki")

    def test_run_sync_job_updates_job_status_and_counters(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
        ]
        job = self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="finance wiki",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=1,
                default_department_scope=["finance"],
                default_async_mode=False,
            ),
            self.user,
        )

        saved_before = self.repository.get_source_sync_job("t1", "feishu", job.job_id)
        self.assertEqual(saved_before.status, "idle")
        self.assertEqual(saved_before.run_count, 0)

        self.service.run_sync_job(job.job_id, self.user)

        saved_after = self.repository.get_source_sync_job("t1", "feishu", job.job_id)
        self.assertEqual(saved_after.status, "idle")
        self.assertEqual(saved_after.run_count, 1)
        self.assertEqual(saved_after.success_count, 1)
        self.assertEqual(saved_after.failure_count, 0)
        self.assertEqual(saved_after.last_run_status, "success")

    def test_run_enabled_sync_jobs_skips_disabled_jobs(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
        ]
        self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="enabled job",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=1,
                default_department_scope=["finance"],
                default_async_mode=False,
                enabled=True,
            ),
            self.user,
        )
        self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="disabled job",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=1,
                default_department_scope=["finance"],
                default_async_mode=False,
                enabled=False,
            ),
            self.user,
        )

        summary = self.service.run_enabled_sync_jobs(self.user)

        self.assertEqual(summary.total_jobs, 2)
        self.assertEqual(summary.succeeded_jobs, 1)
        self.assertEqual(summary.failed_jobs, 0)
        self.assertEqual(summary.skipped_jobs, 1)

    def test_completed_sync_cycle_retires_missing_documents(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
            "https://example.feishu.cn/wiki/wiki_token_2",
        ]
        job = self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="finance wiki",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=10,
                default_department_scope=["finance"],
                default_async_mode=False,
            ),
            self.user,
        )

        self.service.run_sync_job(job.job_id, self.user)
        second_doc = self.repository.find_current_document_by_source_ref("t1", "feishu", "wiki_token_2")
        self.assertIsNotNone(second_doc)
        self.assertTrue(self.service.document_service.source_store.has_source(second_doc.id, second_doc.source_type))

        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
        ]
        self.service.run_sync_job(job.job_id, self.user)

        history = self.repository.list_documents_by_source_ref("t1", "feishu", "wiki_token_2")
        self.assertEqual(len(history), 1)
        retired = history[0]
        self.assertFalse(retired.current)
        self.assertEqual(retired.status, DocumentStatus.RETIRED)
        self.assertFalse(self.service.document_service.source_store.has_source(retired.id, retired.source_type))

    def test_run_enabled_sync_jobs_all_tenants_executes_cross_tenant_jobs(self) -> None:
        self.feishu_client.listed_sources = [
            "https://example.feishu.cn/wiki/wiki_token_1",
        ]
        self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="tenant one job",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=1,
                default_department_scope=["finance"],
                default_async_mode=False,
            ),
            self.user,
        )

        tenant_two_user = UserContext(
            user_id="u2",
            tenant_id="t2",
            department_id="hr",
            role="admin",
            clearance_level=3,
        )
        self.service.upsert_sync_job(
            FeishuSyncJobUpsertRequest(
                name="tenant two job",
                source_root="https://example.feishu.cn/wiki/root_node",
                limit=1,
                default_department_scope=["hr"],
                default_async_mode=False,
            ),
            tenant_two_user,
        )

        summary = self.service.run_enabled_sync_jobs_all_tenants()

        self.assertEqual(summary.total_jobs, 2)
        self.assertEqual(summary.succeeded_jobs, 2)
        self.assertEqual(summary.failed_jobs, 0)
        self.assertEqual(len(self.repository.list_documents("t1")), 1)
        self.assertEqual(len(self.repository.list_documents("t2")), 1)


if __name__ == "__main__":
    unittest.main()
