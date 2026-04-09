import unittest
from datetime import datetime
from pathlib import Path

from app.application.context.builder import AssembledContext
from app.application.evaluation.service import OfflineEvaluationService
from app.domain.citations.services import Citation
from app.domain.evaluation.models import EvalRunResult, EvalRunSummary, EvalSample
from app.domain.prompts.models import PromptTemplate, PromptValidationResult, RenderedPrompt
from app.domain.risk.models import RiskAction
from app.infrastructure.storage.local_eval_dataset_store import LocalEvalDatasetStore
from app.infrastructure.storage.local_eval_run_store import LocalEvalRunStore


class _RetrievalServiceStub:
    def explain(self, user, query, top_k=5):
        from datetime import datetime

        from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
        from app.domain.retrieval.models import RetrievalExplainResponse, RetrievalProfile, RetrievalResult

        now = datetime(2026, 1, 1)
        document = DocumentRecord(
            id="doc_finance",
            tenant_id=user.tenant_id,
            title="报销制度",
            source_type="manual",
            source_uri=None,
            owner_id=user.user_id,
            department_scope=[user.department_id],
            visibility_scope=["tenant"],
            security_level=1,
            version=1,
            status=DocumentStatus.SUCCESS,
            content_hash="hash",
            created_at=now,
            updated_at=now,
            tags=["finance"],
            current=True,
        )
        chunk = DocumentChunk(
            id="chunk_1",
            doc_id="doc_finance",
            tenant_id=user.tenant_id,
            chunk_index=0,
            section_name="审批规则",
            text="报销审批时限为3个工作日。",
            token_count=10,
            security_level=1,
            department_scope=[user.department_id],
            metadata_json={},
        )
        result = RetrievalResult(
            document=document,
            chunk=chunk,
            score=0.9,
            keyword_score=0.8,
            vector_score=0.7,
            matched_terms=["报销", "审批时限"],
            retrieval_sources=["elasticsearch", "pgvector"],
            selection_status="selected",
            selection_reasons=["selected_after_rerank"],
        )
        return RetrievalExplainResponse(
            rewritten_query=query,
            intent="standard_qa",
            profile=RetrievalProfile(name="standard_qa", keyword_weight=0.5, vector_weight=0.5),
            pre_rerank_results=[result],
            results=[result],
        )


class _ContextBuilderStub:
    def build(self, results):
        return AssembledContext(
            results=results,
            citations=[Citation(index=1, doc_id="doc_finance", title="报销制度", section_name="审批规则", version=1)],
            evidence_blocks=["evidence"],
            citation_lines=["[1] 报销制度 / 审批规则 / v1"],
            fallback_evidence_lines=["[1] 报销审批时限为3个工作日。"],
            summary_lines=["[1] 报销制度 / 审批规则：报销审批时限为3个工作日。"],
            citation_text="[1] 报销制度",
        )


class _PromptBuilderStub:
    def build_chat_prompt(self, scene, query, assembled_context, session_summary=""):
        return type(
            "PromptBuild",
            (),
            {
                "scene": scene,
                "query": query,
                "template": PromptTemplate(
                    id="prompt_standard_v1",
                    scene=scene,
                    version=1,
                    name="Standard QA",
                    content="Use evidence only.",
                    output_schema={"sections": "结论,依据,引用来源,限制说明"},
                ),
                "rendered_prompt": RenderedPrompt(instructions="x", input_text="y"),
                "assembled_context": assembled_context,
            },
        )()


class _GenerationServiceStub:
    def generate_chat_answer(self, user, prompt_build, input_risk_action, input_risk_level):
        return type(
            "GenerationResult",
            (),
            {
                "answer": "结论：报销审批时限为3个工作日。\n依据：[1] 报销制度。\n引用来源：[1] 报销制度 / 审批规则 / v1\n限制说明：无。",
                "validation_result": PromptValidationResult(
                    template_id="prompt_standard_v1",
                    template_version=1,
                    valid=True,
                    missing_sections=[],
                    normalized_answer="ok",
                ),
                "action": RiskAction.ALLOW,
            },
        )()


class OfflineEvaluationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = Path("/tmp/secure_rag_gateway_eval_dataset.jsonl")
        self.runs_dir = Path("/tmp/secure_rag_gateway_eval_runs")
        self.dataset_path.unlink(missing_ok=True)
        if self.runs_dir.exists():
            for path in self.runs_dir.glob("*.json"):
                path.unlink()

    def _build_service(self) -> OfflineEvaluationService:
        store = LocalEvalDatasetStore(str(self.dataset_path))
        run_store = LocalEvalRunStore(str(self.runs_dir))
        store.replace_samples(
            [
                EvalSample(
                    id="case_1",
                    query="报销审批时限是什么？",
                    expected_doc_ids=["doc_finance"],
                    expected_titles=["报销制度"],
                    expected_answer_contains=["3个工作日"],
                )
            ]
        )
        return OfflineEvaluationService(
            dataset_store=store,
            run_store=run_store,
            retrieval_service=_RetrievalServiceStub(),
            context_builder=_ContextBuilderStub(),
            prompt_builder=_PromptBuilderStub(),
            generation_service=_GenerationServiceStub(),
        )

    def test_run_returns_summary_and_case_results(self) -> None:
        service = self._build_service()
        result = service.run()

        self.assertEqual(result.dataset_size, 1)
        self.assertEqual(result.summary.total_cases, 1)
        self.assertEqual(result.summary.retrieval_hit_rate, 1.0)
        self.assertEqual(result.summary.answer_match_rate, 1.0)
        self.assertEqual(result.summary.answer_valid_rate, 1.0)
        self.assertEqual(result.cases[0].matched_doc_ids, ["doc_finance"])

    def test_run_persists_and_can_be_listed(self) -> None:
        service = self._build_service()

        run = service.run()
        listed = service.list_runs(limit=10)
        loaded = service.get_run(run.run_id)

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].run_id, run.run_id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["mode"], "offline")
        self.assertEqual(loaded["summary"]["total_cases"], 1)

    def test_run_shadow_returns_comparison_and_persists(self) -> None:
        service = self._build_service()

        run = service.run_shadow()
        loaded = service.get_run(run.run_id)

        self.assertEqual(run.mode, "shadow")
        self.assertEqual(run.dataset_size, 1)
        self.assertEqual(len(run.diffs), 1)
        self.assertEqual(run.diffs[0].sample_id, "case_1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["mode"], "shadow")
        self.assertIn("primary_summary", loaded)
        self.assertIn("shadow_summary", loaded)

    def test_build_trend_summary_detects_regression_alerts(self) -> None:
        service = self._build_service()
        service.run_store.save_run(
            "eval_baseline",
            EvalRunResult(
                run_id="eval_baseline",
                mode="offline",
                dataset_size=1,
                started_at=datetime(2026, 1, 1, 10, 0, 0),
                finished_at=datetime(2026, 1, 1, 10, 1, 0),
                summary=EvalRunSummary(
                    total_cases=1,
                    retrieval_hit_rate=1.0,
                    title_hit_rate=1.0,
                    answer_match_rate=1.0,
                    answer_valid_rate=1.0,
                    average_latency_ms=100.0,
                    average_retrieved_chunks=2.0,
                ),
                cases=[],
            ).model_dump(mode="json"),
        )
        current_run = EvalRunResult(
            run_id="eval_current",
            mode="offline",
            dataset_size=1,
            started_at=datetime(2026, 1, 2, 10, 0, 0),
            finished_at=datetime(2026, 1, 2, 10, 1, 0),
            summary=EvalRunSummary(
                total_cases=1,
                retrieval_hit_rate=0.8,
                title_hit_rate=0.9,
                answer_match_rate=0.9,
                answer_valid_rate=1.0,
                average_latency_ms=250.0,
                average_retrieved_chunks=2.0,
            ),
            cases=[],
        )

        trend = service.build_trend_summary(current_run=current_run)

        self.assertEqual(trend.current_run_id, "eval_current")
        self.assertEqual(trend.baseline_run_id, "eval_baseline")
        self.assertLess(trend.deltas["retrieval_hit_rate"], 0)
        self.assertGreater(trend.deltas["average_latency_ms"], 0)
        self.assertTrue(any(alert.metric == "retrieval_hit_rate" for alert in trend.alerts))
        self.assertTrue(any(alert.metric == "average_latency_ms" for alert in trend.alerts))


if __name__ == "__main__":
    unittest.main()
