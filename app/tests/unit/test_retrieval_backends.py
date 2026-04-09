import unittest
from datetime import datetime

from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.infrastructure.search.elasticsearch import ElasticsearchSearch
from app.infrastructure.vectorstore.pgvector import PGVectorStore


def build_document() -> DocumentRecord:
    now = datetime(2026, 1, 1)
    return DocumentRecord(
        id="doc_1",
        tenant_id="t1",
        title="报销制度",
        source_type="manual",
        source_uri=None,
        owner_id="u1",
        department_scope=["engineering"],
        visibility_scope=["tenant"],
        security_level=1,
        version=1,
        status=DocumentStatus.SUCCESS,
        content_hash="hash",
        created_at=now,
        updated_at=now,
        tags=[],
        current=True,
    )


def build_chunk() -> DocumentChunk:
    return DocumentChunk(
        id="chunk_1",
        doc_id="doc_1",
        tenant_id="t1",
        chunk_index=0,
        section_name="Section 1",
        text="报销审批时限为3个工作日。",
        token_count=6,
        security_level=1,
        department_scope=["engineering"],
        metadata_json={"title": "报销制度"},
    )


class RetrievalBackendTest(unittest.TestCase):
    def test_elasticsearch_backend_returns_keyword_hits(self) -> None:
        backend = ElasticsearchSearch()
        hits = backend.search(
            query="报销审批时限是什么",
            terms=["报销", "审批", "时限"],
            candidates=[(build_document(), build_chunk())],
            top_k=5,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].backend, "elasticsearch")
        self.assertIn("报销", hits[0].matched_terms)

    def test_pgvector_backend_returns_semantic_hits(self) -> None:
        backend = PGVectorStore()
        hits = backend.search(
            query="报销审批时限是什么",
            candidates=[(build_document(), build_chunk())],
            top_k=5,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].backend, "pgvector")
        self.assertGreater(hits[0].score, 0)

    def test_pgvector_sql_and_ddl_include_metadata_filters(self) -> None:
        backend = PGVectorStore()
        from app.application.access.service import build_access_filter
        from app.domain.auth.models import UserContext

        access_filter = build_access_filter(
            UserContext(
                user_id="u1",
                tenant_id="t1",
                department_id="engineering",
                role="employee",
                clearance_level=2,
            )
        )
        ddl = backend.build_table_ddl()
        sql = backend.build_search_sql(access_filter, 5, tag_filters=["finance"], year_filters=[2025])

        self.assertIn("tags JSONB NOT NULL", ddl)
        self.assertIn("created_at TIMESTAMPTZ NOT NULL", ddl)
        self.assertIn("tags ?| %(tag_filters)s::text[]", sql)
        self.assertIn("EXTRACT(YEAR FROM updated_at)", sql)


if __name__ == "__main__":
    unittest.main()
