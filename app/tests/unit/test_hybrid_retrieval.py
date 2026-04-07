import unittest
from datetime import datetime

from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.retrievers import keyword_features, normalize_terms, vector_score


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


class HybridRetrievalTest(unittest.TestCase):
    def test_normalize_terms_expands_chinese_terms(self) -> None:
        terms = normalize_terms("报销审批时限是什么")
        self.assertIn("报销审批时限是什么", terms)
        self.assertIn("报销", terms)
        self.assertIn("审批", terms)

    def test_keyword_features_detect_matches(self) -> None:
        score, matched_terms = keyword_features(["报销", "审批"], build_document(), build_chunk())
        self.assertGreater(score, 0)
        self.assertIn("报销", matched_terms)

    def test_vector_score_prefers_semantic_overlap(self) -> None:
        score = vector_score("报销审批时限是什么", build_document(), build_chunk())
        self.assertGreater(score, 0)


if __name__ == "__main__":
    unittest.main()
