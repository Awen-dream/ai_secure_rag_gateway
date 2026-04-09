import unittest
from datetime import datetime

from app.application.query.planning import QueryPlanningResult
from app.application.query.rewrite import build_query_rewrite_plan, refine_query_rewrite_plan
from app.application.query.understanding import QueryUnderstandingResult
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.services import RetrievalService


class _FakeDocumentService:
    def __init__(self, candidates):
        self._candidates = candidates

    def get_accessible_chunks(self, user):
        return list(self._candidates)


class _FakeKeywordBackend:
    backend_name = "elasticsearch"

    def __init__(self):
        self.last_terms = []
        self.last_tag_filters = []
        self.last_year_filters = []
        self.last_exact_terms = []

    def search(
        self,
        query,
        terms,
        candidates,
        top_k,
        access_filter=None,
        tag_filters=None,
        year_filters=None,
        exact_terms=None,
    ):
        self.last_terms = list(terms)
        self.last_tag_filters = list(tag_filters or [])
        self.last_year_filters = list(year_filters or [])
        self.last_exact_terms = list(exact_terms or [])
        return [
            BackendSearchHit(
                document=document,
                chunk=chunk,
                score=10.0 - index,
                backend=self.backend_name,
                matched_terms=[term for term in terms if term in f"{document.title} {chunk.text}".lower()],
            )
            for index, (document, chunk) in enumerate(candidates[:top_k])
        ]

    def upsert_document(self, document, chunks):
        return {}

    def delete_document(self, document, chunks):
        return {}

    def describe_backend(self):
        return None


class _FakeVectorBackend:
    backend_name = "pgvector"

    def search(self, query, candidates, top_k, access_filter=None, tag_filters=None, year_filters=None):
        return [
            BackendSearchHit(
                document=document,
                chunk=chunk,
                score=0.9 - (index * 0.1),
                backend=self.backend_name,
            )
            for index, (document, chunk) in enumerate(candidates[:top_k])
        ]

    def upsert_document(self, document, chunks):
        return {}

    def delete_document(self, document, chunks):
        return {}

    def describe_backend(self):
        return None


class _FailUnderstandingService:
    def plan(self, query, last_user_query=None, session_summary=None, understanding=None):
        raise AssertionError("retrieval service should reuse precomputed query plan")


def _build_document(doc_id, title, tags, year):
    now = datetime(year, 6, 1)
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
        tags=tags,
        current=True,
    )


def _build_chunk(doc_id, chunk_id, text):
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


class RetrievalQualityTest(unittest.TestCase):
    def test_explain_applies_tag_and_year_filters_and_expanded_terms(self) -> None:
        finance_doc = _build_document("doc_finance", "报销制度", ["finance"], 2025)
        finance_chunk = _build_chunk("doc_finance", "chunk_finance", "报销审批时限为3个工作日。")
        hr_doc = _build_document("doc_hr", "报销制度旧版", ["hr"], 2024)
        hr_chunk = _build_chunk("doc_hr", "chunk_hr", "历史报销审批时限为5个工作日。")

        keyword_backend = _FakeKeywordBackend()
        service = RetrievalService(
            document_service=_FakeDocumentService([(finance_doc, finance_chunk), (hr_doc, hr_chunk)]),
            keyword_backend=keyword_backend,
            vector_backend=_FakeVectorBackend(),
        )
        user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="finance",
            role="admin",
            clearance_level=3,
        )

        explanation = service.explain(user, "请问 2025年 #finance 最新报销审批时限是多少？", top_k=5)

        self.assertEqual(len(explanation.results), 1)
        self.assertEqual(explanation.results[0].document.id, "doc_finance")
        self.assertIn("费用报销", explanation.expanded_terms)
        self.assertEqual(explanation.tag_filters, ["finance"])
        self.assertEqual(explanation.year_filters, [2025])
        self.assertEqual(explanation.rerank_sources, [])
        self.assertIn("审批时限", keyword_backend.last_terms)
        self.assertEqual(keyword_backend.last_tag_filters, ["finance"])
        self.assertEqual(keyword_backend.last_year_filters, [2025])
        self.assertIn("审批时限", keyword_backend.last_exact_terms)

    def test_explain_reuses_precomputed_understanding_and_rewrite_plan(self) -> None:
        finance_doc = _build_document("doc_finance", "报销制度", ["finance"], 2025)
        finance_chunk = _build_chunk("doc_finance", "chunk_finance", "报销审批时限为3个工作日。")

        keyword_backend = _FakeKeywordBackend()
        service = RetrievalService(
            document_service=_FakeDocumentService([(finance_doc, finance_chunk)]),
            keyword_backend=keyword_backend,
            vector_backend=_FakeVectorBackend(),
            query_planning=_FailUnderstandingService(),
        )
        user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="finance",
            role="admin",
            clearance_level=3,
        )
        understanding = QueryUnderstandingResult(
            rewritten_query="报销制度审批时限",
            intent="standard_qa",
            confidence=0.9,
            reasons=["precomputed"],
            source="rule",
            rule_rewritten_query="报销制度审批时限",
            rule_intent="standard_qa",
            rule_confidence=0.9,
            rule_reasons=["precomputed"],
        )
        rewrite_plan = refine_query_rewrite_plan(
            build_query_rewrite_plan("审批时限呢？", last_user_query="报销制度是什么？"),
            understanding.rewritten_query,
        )

        explanation = service.explain(
            user,
            "审批时限呢？",
            top_k=5,
            query_plan=QueryPlanningResult(
                understanding=understanding,
                rewrite_plan=rewrite_plan,
            ),
        )

        self.assertEqual(explanation.rewritten_query, "报销制度审批时限")
        self.assertEqual(explanation.understanding_source, "rule")
        self.assertGreaterEqual(len(explanation.results), 1)


if __name__ == "__main__":
    unittest.main()
