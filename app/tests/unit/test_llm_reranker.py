import unittest
from datetime import datetime
from unittest.mock import patch

from app.application.retrieval.llm_reranker import LLMReranker
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.models import RetrievalResult
from app.infrastructure.llm.openai_client import OpenAIClient


def _build_document(doc_id: str, title: str) -> DocumentRecord:
    now = datetime(2026, 6, 1)
    return DocumentRecord(
        id=doc_id,
        tenant_id="t1",
        title=title,
        source_type="manual",
        source_uri=None,
        owner_id="u1",
        department_scope=["finance"],
        visibility_scope=["tenant"],
        security_level=1,
        version=1,
        status=DocumentStatus.SUCCESS,
        content_hash=f"hash_{doc_id}",
        created_at=now,
        updated_at=now,
        tags=["finance"],
        current=True,
    )


def _build_chunk(doc_id: str, chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        doc_id=doc_id,
        tenant_id="t1",
        chunk_index=0,
        section_name="审批规则",
        text=text,
        token_count=20,
        security_level=1,
        department_scope=["finance"],
        metadata_json={},
    )


class LLMRerankerTest(unittest.TestCase):
    def test_llm_reranker_reorders_candidates_by_model_output(self) -> None:
        reranker = LLMReranker(
            client=OpenAIClient(
                api_key="test-key",
                model="gpt-5.4-mini",
                base_url="https://api.openai.com/v1",
                timeout_seconds=30,
                max_output_tokens=256,
                temperature=0.0,
            ),
            top_n=3,
        )
        results = [
            RetrievalResult(
                document=_build_document("doc_1", "报销制度"),
                chunk=_build_chunk("doc_1", "chunk_1", "报销审批时限为3个工作日。"),
                score=0.90,
                keyword_score=0.8,
                vector_score=0.7,
                matched_terms=["报销", "审批时限"],
                retrieval_sources=["elasticsearch"],
            ),
            RetrievalResult(
                document=_build_document("doc_2", "采购制度"),
                chunk=_build_chunk("doc_2", "chunk_2", "采购审批时限为5个工作日。"),
                score=0.88,
                keyword_score=0.78,
                vector_score=0.68,
                matched_terms=["审批时限"],
                retrieval_sources=["elasticsearch"],
            ),
        ]

        with patch.object(
            OpenAIClient,
            "generate_response",
            return_value='{"ranked_candidates":[{"candidate_id":"R2","score":0.96,"reason":"direct answer"},{"candidate_id":"R1","score":0.72,"reason":"supporting policy"}]}',
        ):
            reranked = reranker.rerank("审批时限是什么？", results)

        self.assertEqual(reranked[0].document.id, "doc_2")
        self.assertEqual(reranked[1].document.id, "doc_1")
        self.assertEqual(reranked[0].rerank_source, "llm")
        self.assertIn("direct answer", reranked[0].rerank_notes[0])


if __name__ == "__main__":
    unittest.main()
