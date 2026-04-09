import unittest
from datetime import datetime

from app.application.context.builder import ContextBuilderService
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.models import RetrievalResult


def _build_document(doc_id, title):
    now = datetime(2026, 6, 1)
    return DocumentRecord(
        id=doc_id,
        tenant_id="t1",
        title=title,
        source_type="manual",
        source_uri="https://example.com/doc",
        owner_id="u1",
        department_scope=["finance"],
        visibility_scope=["tenant"],
        security_level=1,
        version=2,
        status=DocumentStatus.SUCCESS,
        content_hash=f"hash_{doc_id}",
        created_at=now,
        updated_at=now,
        tags=["finance"],
        current=True,
    )


def _build_chunk(doc_id, chunk_id, text, section_name="审批规则"):
    return DocumentChunk(
        id=chunk_id,
        doc_id=doc_id,
        tenant_id="t1",
        chunk_index=0,
        section_name=section_name,
        text=text,
        token_count=20,
        security_level=1,
        department_scope=["finance"],
        metadata_json={},
    )


class ContextBuilderServiceTest(unittest.TestCase):
    def test_build_creates_citations_and_prompt_friendly_blocks(self) -> None:
        document = _build_document("doc_1", "报销制度")
        first_chunk = _build_chunk("doc_1", "chunk_1", "报销审批时限为3个工作日。")
        second_chunk = _build_chunk("doc_1", "chunk_2", "超标费用需要额外审批。", section_name="费用规则")
        results = [
            RetrievalResult(
                document=document,
                chunk=first_chunk,
                score=0.91,
                keyword_score=0.8,
                vector_score=0.7,
                matched_terms=["报销", "审批时限"],
                retrieval_sources=["elasticsearch", "pgvector"],
            ),
            RetrievalResult(
                document=document,
                chunk=second_chunk,
                score=0.84,
                keyword_score=0.7,
                vector_score=0.6,
                matched_terms=["审批"],
                retrieval_sources=["elasticsearch"],
            ),
        ]

        assembled = ContextBuilderService().build(results)

        self.assertEqual(assembled.retrieved_chunks, 2)
        self.assertEqual(len(assembled.citations), 1)
        self.assertEqual(len(assembled.evidence_blocks), 2)
        self.assertEqual(len(assembled.fallback_evidence_lines), 2)
        self.assertEqual(len(assembled.summary_lines), 2)
        self.assertIn("[1] 报销制度", assembled.citation_text)
        self.assertIn("文档：报销制度 v2", assembled.evidence_blocks[0])

    def test_build_compacts_duplicate_chunks(self) -> None:
        document = _build_document("doc_1", "报销制度")
        duplicate_text = "报销审批时限为3个工作日。\n请提供审批单据。"
        results = [
            RetrievalResult(
                document=document,
                chunk=_build_chunk("doc_1", "chunk_1", duplicate_text),
                score=0.91,
                keyword_score=0.8,
                vector_score=0.7,
                matched_terms=["报销", "审批时限"],
                retrieval_sources=["elasticsearch"],
            ),
            RetrievalResult(
                document=document,
                chunk=_build_chunk("doc_1", "chunk_2", duplicate_text.replace("\n", " ")),
                score=0.90,
                keyword_score=0.79,
                vector_score=0.69,
                matched_terms=["报销", "审批时限"],
                retrieval_sources=["pgvector"],
            ),
        ]

        assembled = ContextBuilderService().build(results)

        self.assertEqual(assembled.retrieved_chunks, 1)
        self.assertEqual(len(assembled.evidence_blocks), 1)


if __name__ == "__main__":
    unittest.main()
