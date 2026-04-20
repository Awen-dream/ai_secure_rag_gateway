from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from typing import Any

from app.application.evaluation.engines import NativeEvaluationExecutionEngine
from app.application.context.builder import ContextBuilderService
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuilderService
from app.core.config import settings
from app.domain.auth.models import UserContext
from app.domain.evaluation.models import EvalCaseResult, EvalSample
from app.domain.retrieval.services import RetrievalService
from app.domain.risk.models import RiskAction


@dataclass(frozen=True)
class _LlamaIndexEvaluatorBundle:
    faithfulness_evaluator: Any
    relevancy_evaluator: Any


class LlamaIndexEvaluationExecutionEngine(NativeEvaluationExecutionEngine):
    """Use LlamaIndex evaluators when available, otherwise fall back to the native evaluation engine."""

    engine_name = "llamaindex"

    def run_case(
        self,
        *,
        sample: EvalSample,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilderService,
        prompt_builder: PromptBuilderService,
        generation_service: GenerationService,
    ) -> EvalCaseResult:
        started_at = datetime.utcnow()
        evaluator_bundle = self._build_evaluator_bundle()
        if evaluator_bundle is None:
            return super().run_case(
                sample=sample,
                retrieval_service=retrieval_service,
                context_builder=context_builder,
                prompt_builder=prompt_builder,
                generation_service=generation_service,
            )

        user = UserContext(
            user_id=sample.user_id,
            tenant_id=sample.tenant_id,
            department_id=sample.department_id,
            role=sample.role,
            clearance_level=sample.clearance_level,
        )
        explanation = retrieval_service.explain(user, sample.query, top_k=5)
        assembled_context = context_builder.build(explanation.results)
        prompt_build = prompt_builder.build_chat_prompt(
            scene=sample.scene,
            query=explanation.rewritten_query,
            assembled_context=assembled_context,
            session_summary="",
        )
        generation = generation_service.generate_chat_answer(
            user=user,
            prompt_build=prompt_build,
            input_risk_action=RiskAction.ALLOW,
            input_risk_level="low",
        )

        matched_doc_ids = [
            result.document.id
            for result in explanation.results
            if result.document.id in set(sample.expected_doc_ids)
        ]
        matched_titles = [
            result.document.title
            for result in explanation.results
            if result.document.title in set(sample.expected_titles)
        ]
        answer_contains_expected = all(term in generation.answer for term in sample.expected_answer_contains)
        llamaindex_valid = self._evaluate_with_llamaindex(
            evaluator_bundle=evaluator_bundle,
            query=sample.query,
            answer=generation.answer,
            contexts=[result.chunk.text for result in explanation.results],
        )
        answer_valid = generation.validation_result.valid if llamaindex_valid is None else (
            generation.validation_result.valid and llamaindex_valid
        )
        latency_ms = int((datetime.utcnow() - started_at).total_seconds() * 1000)

        return EvalCaseResult(
            sample_id=sample.id,
            query=sample.query,
            scene=sample.scene,
            hit_expected_doc=bool(matched_doc_ids) if sample.expected_doc_ids else False,
            hit_expected_title=bool(matched_titles) if sample.expected_titles else False,
            answer_contains_expected=answer_contains_expected if sample.expected_answer_contains else False,
            answer_valid=answer_valid,
            matched_doc_ids=matched_doc_ids,
            matched_titles=matched_titles,
            retrieved_chunks=len(explanation.results),
            rewritten_query=explanation.rewritten_query,
            intent=explanation.intent,
            latency_ms=latency_ms,
            validation_missing_sections=generation.validation_result.missing_sections,
            answer_preview=generation.answer[:240],
            citations=[item.title for item in assembled_context.citations],
        )

    def _build_evaluator_bundle(self) -> _LlamaIndexEvaluatorBundle | None:
        if not settings.openai_api_key:
            return None

        try:
            evaluation_module = import_module("llama_index.core.evaluation")
            openai_module = import_module("llama_index.llms.openai")
        except Exception:
            return None

        faithfulness_class = getattr(evaluation_module, "FaithfulnessEvaluator", None)
        relevancy_class = getattr(evaluation_module, "RelevancyEvaluator", None)
        openai_class = getattr(openai_module, "OpenAI", None)
        if not faithfulness_class or not relevancy_class or not openai_class:
            return None

        llm = openai_class(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_base_url,
            timeout=settings.openai_timeout_seconds,
            temperature=settings.openai_temperature,
        )
        return _LlamaIndexEvaluatorBundle(
            faithfulness_evaluator=faithfulness_class(llm=llm),
            relevancy_evaluator=relevancy_class(llm=llm),
        )

    def _evaluate_with_llamaindex(
        self,
        *,
        evaluator_bundle: _LlamaIndexEvaluatorBundle,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> bool | None:
        try:
            faithfulness = self._call_evaluator(
                evaluator_bundle.faithfulness_evaluator,
                query=query,
                answer=answer,
                contexts=contexts,
            )
            relevancy = self._call_evaluator(
                evaluator_bundle.relevancy_evaluator,
                query=query,
                answer=answer,
                contexts=contexts,
            )
        except Exception:
            return None

        faithfulness_pass = self._normalize_eval_result(faithfulness)
        relevancy_pass = self._normalize_eval_result(relevancy)
        if faithfulness_pass is None or relevancy_pass is None:
            return None
        return faithfulness_pass and relevancy_pass

    @staticmethod
    def _call_evaluator(evaluator: Any, *, query: str, answer: str, contexts: list[str]) -> Any:
        payload = {
            "query": query,
            "response": answer,
            "contexts": contexts,
        }
        for method_name in ("evaluate", "evaluate_response"):
            method = getattr(evaluator, method_name, None)
            if callable(method):
                return method(**payload)
        raise RuntimeError("Unsupported LlamaIndex evaluator interface.")

    @staticmethod
    def _normalize_eval_result(result: Any) -> bool | None:
        for attr in ("passing", "passed"):
            value = getattr(result, attr, None)
            if isinstance(value, bool):
                return value
        score = getattr(result, "score", None)
        if isinstance(score, (int, float)):
            return float(score) >= 0.5
        return None
