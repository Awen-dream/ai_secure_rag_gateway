from __future__ import annotations

import shutil
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.domain.evaluation.models import EvalCaseResult, EvalSample
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.domain.retrieval.models import RetrievalResult
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.db.repositories.sqlite import SQLiteRepository
from app.infrastructure.frameworks.llamaindex_eval import (
    LlamaIndexEvaluationExecutionEngine,
    _LlamaIndexEvaluatorBundle,
)
from app.infrastructure.frameworks.llamaindex_ingestion import (
    LlamaIndexDocumentIngestionEngine,
    _LlamaIndexIngestionComponents,
)
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore
from app.infrastructure.vectorstore.pgvector import PGVectorStore


class _FakeLIDocument:
    def __init__(self, text: str, metadata: dict) -> None:
        self.text = text
        self.metadata = metadata


class _FakeNode:
    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self.text = text
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self.text


class _FakeSentenceSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_nodes_from_documents(self, documents: list[_FakeLIDocument]) -> list[_FakeNode]:
        return [
            _FakeNode("第一块内容", {"section_name": "Section A", "heading_path": ["Section A"]}),
            _FakeNode("第二块内容", {"section_name": "Section B", "heading_path": ["Section B"]}),
        ]


class _FakeEvaluator:
    def __init__(self, passing: bool) -> None:
        self.passing = passing

    def evaluate(self, **_: dict) -> SimpleNamespace:
        return SimpleNamespace(passing=self.passing)


class LlamaIndexAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "/tmp/secure_rag_gateway_llamaindex_adapter.db"
        self.staging_dir = Path("/tmp/secure_rag_gateway_llamaindex_staging")
        Path(self.db_path).unlink(missing_ok=True)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        RedisClient.reset_local_state()

        self.repository = SQLiteRepository(self.db_path)
        self.source_store = LocalDocumentSourceStore(str(self.staging_dir))
        self.indexing_service = RetrievalIndexingService(
            keyword_backend=ElasticsearchSearch(index_name="test_chunks", mode="local-fallback"),
            vector_backend=PGVectorStore(table_name="test_vectors", embedding_dimension=64, mode="local-fallback"),
        )
        self.document_service = DocumentService(
            repository=self.repository,
            indexing_service=self.indexing_service,
            source_store=self.source_store,
        )
        self.user = SimpleNamespace(
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

    def test_llamaindex_ingestion_engine_uses_llamaindex_chunking_when_available(self) -> None:
        engine = LlamaIndexDocumentIngestionEngine()
        document = self.document_service.upload_document_file(
            payload=DocumentUploadRequest(
                title="policy.html",
                content="",
                source_type="html",
                department_scope=["engineering"],
                security_level=1,
            ),
            user=self.user,
            file_bytes="<h1>报销制度</h1><p>审批时限为3个工作日。</p>".encode("utf-8"),
        )

        with patch.object(
            engine,
            "_load_components",
            return_value=_LlamaIndexIngestionComponents(
                document_class=_FakeLIDocument,
                sentence_splitter_class=_FakeSentenceSplitter,
                markdown_parser_class=None,
            ),
        ), patch(
            "app.application.ingestion.engines.NativeDocumentIngestionEngine.process_document",
            side_effect=AssertionError("native fallback should not run"),
        ):
            processed = engine.process_document(
                repository=self.repository,
                indexing_service=self.indexing_service,
                source_store=self.source_store,
                retrieval_cache=None,
                doc_id=document.id,
            )

        chunks = self.repository.list_chunks_for_document(document.id)
        self.assertEqual(processed.status, DocumentStatus.SUCCESS)
        self.assertEqual([chunk.text for chunk in chunks], ["第一块内容", "第二块内容"])
        self.assertEqual(chunks[0].metadata_json["node_parser"], "llamaindex")

    def test_llamaindex_evaluation_engine_uses_framework_evaluators_when_available(self) -> None:
        engine = LlamaIndexEvaluationExecutionEngine()
        sample = EvalSample(
            id="case_1",
            query="报销审批时限是什么？",
            expected_doc_ids=["doc_finance"],
            expected_titles=["报销制度"],
            expected_answer_contains=["3个工作日"],
        )

        retrieval_result = RetrievalResult(
            document=DocumentRecord(
                id="doc_finance",
                tenant_id="eval",
                title="报销制度",
                source_type="manual",
                source_uri=None,
                owner_id="eval_user",
                department_scope=["engineering"],
                visibility_scope=["tenant"],
                security_level=1,
                content_hash="hash",
                created_at=datetime(2026, 1, 1),
                updated_at=datetime(2026, 1, 1),
                tags=[],
                current=True,
            ),
            chunk=DocumentChunk(
                id="chunk_1",
                doc_id="doc_finance",
                tenant_id="eval",
                chunk_index=0,
                section_name="Document",
                text="审批时限为3个工作日。",
                token_count=6,
                security_level=1,
                department_scope=["engineering"],
                metadata_json={},
            ),
            score=0.9,
            keyword_score=0.8,
            vector_score=0.7,
            retrieval_sources=["elasticsearch"],
        )

        retrieval_service = SimpleNamespace(
            explain=lambda *_args, **_kwargs: SimpleNamespace(
                results=[retrieval_result],
                rewritten_query=sample.query,
                intent="standard_qa",
            )
        )
        context_builder = SimpleNamespace(
            build=lambda results: SimpleNamespace(citations=[SimpleNamespace(title=result.document.title) for result in results])
        )
        prompt_builder = SimpleNamespace(build_chat_prompt=lambda **_kwargs: SimpleNamespace())
        generation_service = SimpleNamespace(
            generate_chat_answer=lambda **_kwargs: SimpleNamespace(
                answer="结论：审批时限为3个工作日。",
                validation_result=SimpleNamespace(valid=True, missing_sections=[]),
            )
        )

        with patch.object(
            engine,
            "_build_evaluator_bundle",
            return_value=_LlamaIndexEvaluatorBundle(
                faithfulness_evaluator=_FakeEvaluator(True),
                relevancy_evaluator=_FakeEvaluator(False),
            ),
        ), patch(
            "app.application.evaluation.engines.NativeEvaluationExecutionEngine.run_case",
            side_effect=AssertionError("native fallback should not run"),
        ):
            result = engine.run_case(
                sample=sample,
                retrieval_service=retrieval_service,
                context_builder=context_builder,
                prompt_builder=prompt_builder,
                generation_service=generation_service,
            )

        self.assertIsInstance(result, EvalCaseResult)
        self.assertFalse(result.answer_valid)
        self.assertEqual(result.matched_doc_ids, ["doc_finance"])
