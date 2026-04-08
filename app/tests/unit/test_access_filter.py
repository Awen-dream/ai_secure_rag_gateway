import unittest
from datetime import datetime

from app.domain.auth.filter_builder import AccessFilter, build_access_filter
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus


def build_user() -> UserContext:
    return UserContext(
        user_id="u1",
        tenant_id="t1",
        department_id="engineering",
        role="employee",
        clearance_level=2,
    )


def build_document(
    visibility_scope=None,
    owner_id: str = "owner_1",
    department_scope=None,
    security_level: int = 1,
) -> DocumentRecord:
    now = datetime(2026, 1, 1)
    return DocumentRecord(
        id="doc_1",
        tenant_id="t1",
        title="报销制度",
        source_type="manual",
        source_uri=None,
        owner_id=owner_id,
        department_scope=department_scope or ["engineering"],
        visibility_scope=visibility_scope or ["tenant"],
        security_level=security_level,
        version=1,
        status=DocumentStatus.SUCCESS,
        content_hash="hash",
        created_at=now,
        updated_at=now,
        tags=[],
        current=True,
    )


def build_chunk(role_scope=None, department_scope=None, security_level: int = 1) -> DocumentChunk:
    return DocumentChunk(
        id="chunk_1",
        doc_id="doc_1",
        tenant_id="t1",
        chunk_index=0,
        section_name="Section 1",
        text="报销审批时限为3个工作日。",
        token_count=6,
        security_level=security_level,
        department_scope=department_scope or ["engineering"],
        role_scope=role_scope or [],
        metadata_json={"title": "报销制度"},
    )


class AccessFilterTest(unittest.TestCase):
    def test_owner_visibility_allows_owner_only(self) -> None:
        access_filter = build_access_filter(build_user())

        self.assertTrue(access_filter.allows_document(build_document(visibility_scope=["owner"], owner_id="u1")))
        self.assertFalse(access_filter.allows_document(build_document(visibility_scope=["owner"], owner_id="u2")))

    def test_department_visibility_requires_department_match(self) -> None:
        access_filter = build_access_filter(build_user())

        self.assertTrue(access_filter.allows_document(build_document(visibility_scope=["department"])))
        self.assertFalse(
            access_filter.allows_document(
                build_document(visibility_scope=["department"], department_scope=["finance"])
            )
        )

    def test_chunk_filter_respects_role_and_security(self) -> None:
        access_filter = build_access_filter(build_user())

        self.assertTrue(access_filter.allows_chunk(build_chunk(role_scope=["employee"], security_level=2)))
        self.assertFalse(access_filter.allows_chunk(build_chunk(role_scope=["manager"])))
        self.assertFalse(access_filter.allows_chunk(build_chunk(security_level=3)))

    def test_elasticsearch_and_pgvector_filter_artifacts_include_access_constraints(self) -> None:
        access_filter = AccessFilter(
            tenant_id="t1",
            user_id="u1",
            department_id="engineering",
            role="employee",
            max_security_level=2,
        )

        es_filters = access_filter.build_elasticsearch_filters(["chunk_1"])
        sql_clause = access_filter.build_pgvector_where_clause()
        sql_params = access_filter.build_pgvector_params(["chunk_1"])

        self.assertEqual(es_filters[0], {"term": {"tenant_id": "t1"}})
        self.assertIn({"terms": {"chunk_id": ["chunk_1"]}}, es_filters)
        self.assertIn("owner_id = %(user_id)s", sql_clause)
        self.assertIn("visibility_scope ? 'department'", sql_clause)
        self.assertEqual(sql_params["department_id"], "engineering")


if __name__ == "__main__":
    unittest.main()
