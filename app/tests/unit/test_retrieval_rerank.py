import unittest
from datetime import datetime

from app.application.query.planning import QueryPlanningResult
from app.application.query.rewrite import build_query_rewrite_plan, refine_query_rewrite_plan
from app.application.query.understanding import QueryUnderstandingResult
from app.application.retrieval.rerank import RetrievalRerankService
from app.application.retrieval.planning import RecallPlanningService
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.rerankers import RetrievalReranker


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


class RetrievalRerankServiceTest(unittest.TestCase):
    def test_build_and_rerank_results(self) -> None:
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
            build_query_rewrite_plan("请问 2025年 #finance 最新审批时限呢？", last_user_query="报销制度是什么？"),
            understanding.rewritten_query,
        )
        recall_plan = RecallPlanningService().plan(
            QueryPlanningResult(
                understanding=understanding,
                rewrite_plan=rewrite_plan,
            )
        )

        document = _build_document("doc_1", "报销制度", ["finance"], 2025)
        chunk = _build_chunk("doc_1", "chunk_1", "报销审批时限为3个工作日。")
        keyword_hits = [
            BackendSearchHit(
                document=document,
                chunk=chunk,
                score=8.0,
                backend="elasticsearch",
                matched_terms=["报销", "审批时限"],
            )
        ]
        vector_hits = [
            BackendSearchHit(
                document=document,
                chunk=chunk,
                score=0.88,
                backend="pgvector",
            )
        ]

        service = RetrievalRerankService()
        rerank_candidates = service.build_rerank_candidates(keyword_hits, vector_hits, recall_plan)
        execution = service.execute_rerank(rerank_candidates, recall_plan)
        selected = execution.results

        self.assertEqual(len(rerank_candidates), 1)
        self.assertEqual(len(execution.pre_rerank_results), 1)
        self.assertEqual(len(selected), 1)
        self.assertIn("elasticsearch", selected[0].retrieval_sources)
        self.assertIn("pgvector", selected[0].retrieval_sources)
        self.assertGreater(selected[0].score, 0)
        self.assertEqual(selected[0].selection_status, "selected")

    def test_rerank_limits_duplicate_chunks_from_same_document(self) -> None:
        understanding = QueryUnderstandingResult(
            rewritten_query="报销制度审批时限",
            intent="exact_lookup",
            confidence=0.9,
            reasons=["precomputed"],
            source="rule",
            rule_rewritten_query="报销制度审批时限",
            rule_intent="exact_lookup",
            rule_confidence=0.9,
            rule_reasons=["precomputed"],
        )
        recall_plan = RecallPlanningService().plan(
            QueryPlanningResult(
                understanding=understanding,
                rewrite_plan=refine_query_rewrite_plan(
                    build_query_rewrite_plan("报销制度审批时限"),
                    understanding.rewritten_query,
                ),
            )
        )

        doc_one = _build_document("doc_1", "报销制度", ["finance"], 2025)
        doc_two = _build_document("doc_2", "采购制度", ["finance"], 2025)
        results = [
            BackendSearchHit(document=doc_one, chunk=_build_chunk("doc_1", "chunk_1", "报销审批时限为3个工作日。"), score=8.0, backend="elasticsearch", matched_terms=["审批时限"]),
            BackendSearchHit(document=doc_one, chunk=_build_chunk("doc_1", "chunk_2", "报销审批需要财务复核。"), score=7.5, backend="elasticsearch", matched_terms=["审批"]),
            BackendSearchHit(document=doc_two, chunk=_build_chunk("doc_2", "chunk_3", "采购审批时限为5个工作日。"), score=7.0, backend="elasticsearch", matched_terms=["审批时限"]),
        ]

        service = RetrievalRerankService()
        candidates = service.build_rerank_candidates(results, [], recall_plan)
        selected = service.rerank_results(candidates, recall_plan)

        self.assertEqual(selected[0].document.id, "doc_1")
        self.assertEqual(selected[1].document.id, "doc_2")

    def test_execute_rerank_marks_drop_reasons(self) -> None:
        class _DropSecondReranker(RetrievalReranker):
            def rerank(self, query, results):
                return results[:1]

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
        recall_plan = RecallPlanningService().plan(
            QueryPlanningResult(
                understanding=understanding,
                rewrite_plan=refine_query_rewrite_plan(
                    build_query_rewrite_plan("报销制度审批时限"),
                    understanding.rewritten_query,
                ),
            )
        )
        doc_one = _build_document("doc_1", "报销制度", ["finance"], 2025)
        doc_two = _build_document("doc_2", "采购制度", ["finance"], 2025)
        candidates = [
            BackendSearchHit(
                document=doc_one,
                chunk=_build_chunk("doc_1", "chunk_1", "报销审批时限为3个工作日。"),
                score=8.0,
                backend="elasticsearch",
                matched_terms=["审批时限"],
            ),
            BackendSearchHit(
                document=doc_two,
                chunk=_build_chunk("doc_2", "chunk_2", "采购审批时限为5个工作日。"),
                score=7.0,
                backend="elasticsearch",
                matched_terms=["审批时限"],
            ),
        ]

        service = RetrievalRerankService(reranker=_DropSecondReranker())
        execution = service.execute_rerank(service.build_rerank_candidates(candidates, [], recall_plan), recall_plan)

        dropped = [item for item in execution.pre_rerank_results if item.selection_status == "dropped"]
        self.assertEqual(len(dropped), 1)
        self.assertIn("dropped_by_reranker", dropped[0].selection_reasons)


if __name__ == "__main__":
    unittest.main()
