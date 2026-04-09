import shutil
import unittest
from pathlib import Path

from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentStatus
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore


class DocumentIngestionOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_ingestion_orchestrator.db"
        self.staging_dir = Path("/tmp/secure_rag_gateway_ingestion_staging")
        Path(self.db_path).unlink(missing_ok=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        RedisClient.reset_local_state()

        repository = SQLiteRepository(self.db_path)
        source_store = LocalDocumentSourceStore(str(self.staging_dir))
        indexing_service = RetrievalIndexingService(
            keyword_backend=ElasticsearchSearch(index_name="test_chunks", mode="local-fallback"),
            vector_backend=PGVectorStore(table_name="test_vectors", embedding_dimension=64, mode="local-fallback"),
        )
        retrieval_cache = RetrievalCache(redis_client=RedisClient(), ttl_seconds=60)
        orchestrator = DocumentIngestionOrchestrator(
            repository=repository,
            indexing_service=indexing_service,
            source_store=source_store,
            retrieval_cache=retrieval_cache,
        )

        self.repository = repository
        self.orchestrator = orchestrator
        self.retrieval_cache = retrieval_cache
        self.service = DocumentService(
            repository=repository,
            indexing_service=indexing_service,
            source_store=source_store,
            ingestion_orchestrator=orchestrator,
        )
        self.user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="engineering",
            role="employee",
            clearance_level=2,
        )

    def tearDown(self) -> None:
        Path(self.db_path).unlink(missing_ok=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        RedisClient.reset_local_state()

    def test_process_document_moves_pending_upload_to_success(self) -> None:
        document = self.service.upload_document_file(
            payload=DocumentUploadRequest(
                title="policy.html",
                content="",
                source_type="html",
                department_scope=["engineering"],
                security_level=1,
                async_mode=True,
            ),
            user=self.user,
            file_bytes="<h1>报销制度</h1><p>审批时限为3个工作日。</p>".encode("utf-8"),
            process_async=True,
        )

        self.assertEqual(document.status, DocumentStatus.PENDING)
        self.assertEqual(self.repository.list_chunks_for_document(document.id), [])

        processed = self.orchestrator.process_document(document.id)

        self.assertEqual(processed.status, DocumentStatus.SUCCESS)
        self.assertTrue(processed.current)
        self.assertIsNone(processed.last_error)
        self.assertGreaterEqual(len(self.repository.list_chunks_for_document(document.id)), 1)

    def test_process_document_invalidates_stale_retrieval_cache(self) -> None:
        document = self.service.upload_document_file(
            payload=DocumentUploadRequest(
                title="policy.html",
                content="",
                source_type="html",
                department_scope=["engineering"],
                security_level=1,
                async_mode=True,
            ),
            user=self.user,
            file_bytes="<h1>报销制度</h1><p>审批时限为3个工作日。</p>".encode("utf-8"),
            process_async=True,
        )

        self.retrieval_cache.set_results(self.user, "报销审批时限是什么？", 5, [])
        self.assertEqual(self.retrieval_cache.get_results(self.user, "报销审批时限是什么？", 5), [])

        self.orchestrator.process_document(document.id)

        self.assertIsNone(self.retrieval_cache.get_results(self.user, "报销审批时限是什么？", 5))

    def test_failed_document_can_be_reset_to_pending_for_retry(self) -> None:
        document = self.service.upload_document_file(
            payload=DocumentUploadRequest(
                title="broken.pdf",
                content="",
                source_type="pdf",
                department_scope=["engineering"],
                security_level=1,
                async_mode=True,
            ),
            user=self.user,
            file_bytes=b"not-a-real-pdf",
            process_async=True,
        )

        failed = self.orchestrator.process_document(document.id)
        self.assertEqual(failed.status, DocumentStatus.FAILED)
        self.assertTrue(failed.last_error)

        retried = self.orchestrator.retry_document(document.id)
        self.assertEqual(retried.status, DocumentStatus.PENDING)
        self.assertIsNone(retried.last_error)


if __name__ == "__main__":
    unittest.main()
