from __future__ import annotations

from dataclasses import dataclass

from app.application.prompting.builder import PromptBuildResult
from app.domain.auth.models import UserContext
from app.domain.prompts.models import PromptValidationResult
from app.domain.prompts.services import PromptService
from app.domain.risk.models import OutputGuardResult, RiskAction
from app.domain.risk.output_guard import OutputGuard
from app.infrastructure.llm.openai_client import OpenAIClient


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
        prompt_service: PromptService,
        output_guard: OutputGuard,
        openai_client: OpenAIClient,
    ) -> None:
        self.prompt_service = prompt_service
        self.output_guard = output_guard
        self.openai_client = openai_client

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
        validation = self.prompt_service.validate_output(prompt_build.scene, guard_result.answer)
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
            return (
                "结论：当前请求触发了安全策略，平台已拒绝回答。\n"
                "依据：问题中包含高风险指令或疑似 Prompt Injection 特征。"
            )
        if not assembled_context.results:
            if risk_action == RiskAction.CITATIONS_ONLY:
                return "结论：根据当前已授权资料无法确认。\n依据：高敏部门在无证据命中时只返回保守结果。"
            return "结论：根据当前已授权资料无法确认。\n依据：检索范围内没有找到足够证据。"

        if self.openai_client.can_execute():
            try:
                return self.openai_client.generate_response(
                    instructions=prompt_build.rendered_prompt.instructions,
                    input_text=prompt_build.rendered_prompt.input_text,
                )
            except Exception:
                pass

        return (
            "结论：已基于授权知识范围给出回答。\n"
            f"依据：问题“{prompt_build.query}”命中了 {assembled_context.retrieved_chunks} 个权限内知识片段，模板策略为 {prompt_build.template.name}。\n"
            + "\n".join(assembled_context.fallback_evidence_lines)
            + f"\n引用来源：{assembled_context.citation_text}"
        )

    @staticmethod
    def _max_risk_level(current: str, candidate: str) -> str:
        ranking = {"low": 1, "medium": 2, "high": 3}
        return candidate if ranking.get(candidate, 0) > ranking.get(current, 0) else current
