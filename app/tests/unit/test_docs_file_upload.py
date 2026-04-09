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


class DocsFileUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_docs_file_upload.db"
        Path(self.db_path).unlink(missing_ok=True)

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
            get_audit_service,
            get_openai_client,
            get_query_understanding_service,
            get_query_planning_service,
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

    def test_upload_html_file_extracts_text_before_indexing(self) -> None:
        response = self.client.post(
            "/api/v1/docs/upload-file",
            files={
                "file": (
                    "policy.html",
                    "<h1>报销制度</h1><p>审批时限为3个工作日。</p>".encode("utf-8"),
                    "text/html",
                )
            },
            data={"security_level": "1", "department_scope": "engineering"},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        document = response.json()
        self.assertEqual(document["source_type"], "html")

        detail = self.client.post(
            "/api/v1/chat/query",
            json={"query": "报销审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(detail.status_code, 200)
        self.assertIn("审批时限为3个工作日", detail.json()["answer"])

    def test_upload_docx_file_extracts_text_before_indexing(self) -> None:
        response = self.client.post(
            "/api/v1/docs/upload-file",
            files={
                "file": (
                    "flow.docx",
                    build_docx_bytes("采购流程", "审批时限为2个工作日。"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            data={"security_level": "1", "department_scope": "engineering"},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        document = response.json()
        self.assertEqual(document["source_type"], "docx")

        detail = self.client.post(
            "/api/v1/chat/query",
            json={"query": "采购审批时限是什么？"},
            headers=self.headers,
        )
        self.assertEqual(detail.status_code, 200)
        self.assertIn("审批时限为2个工作日", detail.json()["answer"])


if __name__ == "__main__":
    unittest.main()
