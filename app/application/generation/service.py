from __future__ import annotations

from dataclasses import dataclass

from app.application.prompting.builder import PromptBuildResult
from app.domain.auth.models import UserContext
from app.domain.prompts.models import PromptValidationResult
from app.domain.prompts.template_service import PromptTemplateService
from app.domain.risk.models import OutputGuardResult, RiskAction
from app.domain.risk.output_guard import OutputGuard
from app.infrastructure.llm.base import LLMClient


@dataclass(frozen=True)
class GenerationResult:
    raw_answer: str
    answer: str
    action: RiskAction
    risk_level: str
    guard_result: OutputGuardResult
    validation_result: PromptValidationResult


class GenerationService:
    """Own the generation, fallback, output guard, and validation stages."""

    def __init__(
        self,
        prompt_template_service: PromptTemplateService,
        output_guard: OutputGuard,
        llm_client: LLMClient,
    ) -> None:
        self.prompt_template_service = prompt_template_service
        self.output_guard = output_guard
        self.llm_client = llm_client

    def generate_chat_answer(
        self,
        user: UserContext,
        prompt_build: PromptBuildResult,
        input_risk_action: RiskAction,
        input_risk_level: str,
    ) -> GenerationResult:
        raw_answer = self._generate_raw_answer(prompt_build, input_risk_action)
        guard_result = self.output_guard.apply(
            user=user,
            answer=raw_answer,
            citations=prompt_build.assembled_context.citations,
            risk_action=input_risk_action,
        )
        validation = self.prompt_template_service.validate_output(prompt_build.scene, guard_result.answer)
        return GenerationResult(
            raw_answer=raw_answer,
            answer=validation.normalized_answer,
            action=guard_result.action,
            risk_level=self._max_risk_level(input_risk_level, guard_result.risk_level),
            guard_result=guard_result,
            validation_result=validation,
        )

    def _generate_raw_answer(self, prompt_build: PromptBuildResult, risk_action: RiskAction) -> str:
        assembled_context = prompt_build.assembled_context
        if risk_action == RiskAction.REFUSE:
            return self._build_refusal_answer()
        if not assembled_context.results:
            if risk_action == RiskAction.CITATIONS_ONLY:
                return self._build_no_evidence_answer("高敏部门在无证据命中时只返回保守结果。")
            return self._build_no_evidence_answer("检索范围内没有找到足够证据。")

        if self.llm_client.can_execute():
            try:
                return self.llm_client.generate_response(
                    instructions=prompt_build.rendered_prompt.instructions,
                    input_text=prompt_build.rendered_prompt.input_text,
                )
            except Exception:
                pass

        return self._build_fallback_answer(prompt_build)

    @staticmethod
    def _build_refusal_answer() -> str:
        return "\n".join(
            [
                "结论：当前请求触发了安全策略，平台已拒绝回答。",
                "依据：问题中包含高风险指令或疑似 Prompt Injection 特征。",
                "引用来源：无。",
                "限制说明：该请求因安全策略被拦截，未进入正常知识检索与生成流程。",
            ]
        )

    @staticmethod
    def _build_no_evidence_answer(reason: str) -> str:
        return "\n".join(
            [
                "结论：根据当前已授权资料无法确认。",
                f"依据：{reason}",
                "引用来源：无。",
                "限制说明：当前授权范围内未命中可支撑回答的证据片段。",
            ]
        )

    @staticmethod
    def _build_fallback_answer(prompt_build: PromptBuildResult) -> str:
        assembled_context = prompt_build.assembled_context
        top_snippet = assembled_context.fallback_evidence_lines[0].split("] ", 1)[-1] if assembled_context.fallback_evidence_lines else "已命中相关授权证据。"
        basis_lines = assembled_context.summary_lines[:3] or assembled_context.fallback_evidence_lines[:3]
        basis_block = "\n".join(basis_lines)
        citation_block = "；".join(assembled_context.citation_lines[:3]) if assembled_context.citation_lines else "无。"
        limit_parts = [
            f"回答基于当前命中的 {assembled_context.retrieved_chunks} 个授权片段",
            f"{len(assembled_context.citations)} 份文档来源",
        ]
        if assembled_context.retrieved_chunks <= 1:
            limit_parts.append("证据覆盖较窄")
        return "\n".join(
            [
                f"结论：根据当前已授权资料，{top_snippet}",
                f"依据：\n{basis_block}",
                f"引用来源：{citation_block}",
                f"限制说明：{'；'.join(limit_parts)}，建议结合原文进一步确认。",
            ]
        )

    @staticmethod
    def _max_risk_level(current: str, candidate: str) -> str:
        ranking = {"low": 1, "medium": 2, "high": 3}
        return candidate if ranking.get(candidate, 0) > ranking.get(current, 0) else current
