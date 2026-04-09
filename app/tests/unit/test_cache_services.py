import unittest
from datetime import datetime

from app.application.session.cache import SessionCache
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.auth.models import UserContext
from app.domain.chat.models import ChatSession, SessionStatus
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.models import RetrievalResult
from app.domain.risk.rate_limit import RateLimitService
from app.infrastructure.cache.redis_client import RedisClient
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue


def build_user() -> UserContext:
    return UserContext(
        user_id="u1",
        tenant_id="t1",
        department_id="engineering",
        role="employee",
        clearance_level=2,
    )


def build_result() -> RetrievalResult:
    now = datetime(2026, 1, 1)
    return RetrievalResult(
        document=DocumentRecord(
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
        ),
        chunk=DocumentChunk(
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
        ),
        score=0.91,
        keyword_score=1.0,
        vector_score=0.88,
        retrieval_sources=["elasticsearch", "pgvector"],
    )


class CacheServicesTest(unittest.TestCase):
    def test_session_cache_round_trip_with_local_fallback(self) -> None:
        redis_client = RedisClient()
        cache = SessionCache(redis_client=redis_client, ttl_seconds=60)
        session = ChatSession(
            id="session_1",
            tenant_id="t1",
            user_id="u1",
            scene="standard_qa",
            status=SessionStatus.ACTIVE,
            summary="assistant:审批时限为3个工作日。",
            created_at=datetime(2026, 1, 1),
            updated_at=datetime(2026, 1, 1),
        )

        cache.set_session(session)
        cache.set_summary(session.id, session.summary)

        self.assertEqual(cache.get_session(session.id).id, "session_1")
        self.assertEqual(cache.get_summary(session.id), "assistant:审批时限为3个工作日。")

    def test_retrieval_cache_round_trip_with_local_fallback(self) -> None:
        redis_client = RedisClient()
        cache = RetrievalCache(redis_client=redis_client, ttl_seconds=60)
        user = build_user()
        result = build_result()

        cache.set_results(user, "报销审批时限是什么", 5, [result])
        cached = cache.get_results(user, "报销审批时限是什么", 5)

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0].document.title, "报销制度")

    def test_rate_limit_service_blocks_after_threshold(self) -> None:
        service = RateLimitService(redis_client=RedisClient(), window_seconds=60, max_requests=2)

        self.assertEqual(service.check_user("u-limit"), (True, 1))
        self.assertEqual(service.check_user("u-limit"), (True, 2))
        self.assertEqual(service.check_user("u-limit"), (False, 3))

    def test_document_ingestion_queue_round_trip_with_local_fallback(self) -> None:
        queue = DocumentIngestionTaskQueue(redis_client=RedisClient(), queue_name="queue:test_document_ingestion")

        receipt = queue.enqueue_document("doc_1")
        self.assertEqual(receipt["doc_id"], "doc_1")
        self.assertEqual(receipt["status"], "queued")
        self.assertEqual(queue.queue_depth(), 1)

        task = queue.dequeue_document()
        self.assertIsNotNone(task)
        self.assertEqual(task["doc_id"], "doc_1")
        self.assertEqual(queue.queue_depth(), 0)


if __name__ == "__main__":
    unittest.main()
