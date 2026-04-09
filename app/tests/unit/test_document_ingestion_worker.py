import io
import os
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


def build_docx_bytes(*paragraphs: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        document_xml = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>',
        ]
        for paragraph in paragraphs:
            document_xml.append(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>")
        document_xml.append("</w:body></w:document>")
        archive.writestr("word/document.xml", "".join(document_xml))
    return buffer.getvalue()


class DocumentIngestionWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_ingestion_worker.db"
        self.staging_dir = "/tmp/secure_rag_gateway_ingestion_worker_staging"
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.staging_dir).mkdir(parents=True, exist_ok=True)

        os.environ["APP_REPOSITORY_BACKEND"] = "sqlite"
        os.environ["APP_SQLITE_PATH"] = self.db_path
        os.environ["APP_REDIS_MODE"] = "local-fallback"
        os.environ["APP_RATE_LIMIT_MAX_REQUESTS"] = "30"
        os.environ["APP_DOCUMENT_STAGING_DIR"] = self.staging_dir
        os.environ["APP_DOCUMENT_INGESTION_QUEUE_NAME"] = "queue:test_document_ingestion_worker"
        os.environ["OPENAI_API_KEY"] = ""

        from app.core.config import settings
        from app.infrastructure.cache.redis_client import RedisClient

        settings.repository_backend = "sqlite"
        settings.sqlite_path = self.db_path
        settings.redis_mode = "local-fallback"
        settings.rate_limit_max_requests = 30
        settings.document_staging_dir = self.staging_dir
        settings.document_ingestion_queue_name = "queue:test_document_ingestion_worker"
        settings.openai_api_key = None
        RedisClient.reset_local_state()

        from app.api.deps import (
            get_audit_service,
            get_chat_service,
            get_context_builder_service,
            get_document_ingestion_orchestrator,
            get_document_ingestion_worker,
            get_document_service,
            get_document_source_store,
            get_document_task_queue,
            get_indexing_service,
            get_keyword_backend,
            get_openai_client,
            get_output_guard,
            get_policy_engine,
            get_prompt_builder_service,
            get_generation_service,
            get_prompt_template_service,
            get_query_planning_service,
            get_recall_planning_service,
            get_retrieval_rerank_service,
            get_query_understanding_service,
            get_prompt_service,
            get_prompt_template_service,
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
            get_document_task_queue,
            get_keyword_backend,
            get_vector_backend,
            get_document_source_store,
            get_indexing_service,
            get_document_ingestion_orchestrator,
            get_document_ingestion_worker,
            get_document_service,
            get_prompt_service,
            get_prompt_template_service,
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
        self.get_document_ingestion_worker = get_document_ingestion_worker

    def tearDown(self) -> None:
        Path(self.db_path).unlink(missing_ok=True)
        for child in Path(self.staging_dir).glob("*"):
            child.unlink(missing_ok=True)
        Path(self.staging_dir).rmdir()

    def test_async_upload_queues_work_until_worker_processes_document(self) -> None:
        upload = self.client.post(
            "/api/v1/docs/upload-file",
            files={
                "file": (
                    "policy.docx",
                    build_docx_bytes("报销制度", "审批时限为3个工作日。"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            data={"security_level": "1", "department_scope": "engineering", "async_mode": "true"},
            headers=self.headers,
        )

        self.assertEqual(upload.status_code, 200)
        document = upload.json()
        self.assertEqual(document["status"], "pending")

        immediate = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(immediate.status_code, 200)
        self.assertIn("无法确认", immediate.json()["answer"])

        processed = self.get_document_ingestion_worker().process_once()
        self.assertIsNotNone(processed)
        self.assertEqual(processed["status"], "success")

        refreshed = self.client.get(f"/api/v1/docs/{document['id']}", headers=self.headers)
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.json()["status"], "success")

        queried = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(queried.status_code, 200)
        self.assertIn("审批时限为3个工作日", queried.json()["answer"])


if __name__ == "__main__":
    unittest.main()
