import unittest

from app.application.context.builder import AssembledContext
from app.application.generation.service import GenerationService
from app.application.prompting.builder import PromptBuildResult
from app.domain.citations.services import Citation
from app.domain.auth.models import UserContext
from app.domain.prompts.models import PromptTemplate, PromptValidationResult, RenderedPrompt
from app.domain.risk.output_guard import OutputGuard
from app.domain.risk.models import RiskAction
from app.infrastructure.llm.openai_client import OpenAIClient


class _PromptServiceStub:
    def validate_output(self, scene: str, answer: str):
        return PromptValidationResult(
            template_id="prompt_standard_v1",
            template_version=1,
            valid=True,
            missing_sections=[],
            normalized_answer=answer,
        )


class GenerationServiceTest(unittest.TestCase):
    def test_generate_chat_answer_uses_fallback_when_no_results(self) -> None:
        service = GenerationService(
            prompt_template_service=_PromptServiceStub(),
            output_guard=OutputGuard(),
            llm_client=OpenAIClient(
                api_key=None,
                model="gpt-5.4-mini",
                base_url="https://api.openai.com/v1",
                timeout_seconds=30,
                max_output_tokens=256,
                temperature=0.0,
            ),
        )
        built = PromptBuildResult(
            scene="standard_qa",
            query="报销审批时限是什么？",
            template=PromptTemplate(
                id="prompt_standard_v1",
                scene="standard_qa",
                version=1,
                name="Standard QA",
                content="Use evidence only.",
                output_schema={"sections": "结论,依据,引用来源,限制说明"},
            ),
            rendered_prompt=RenderedPrompt(instructions="x", input_text="y"),
            assembled_context=AssembledContext(results=[], citations=[]),
        )
        user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="engineering",
            role="employee",
            clearance_level=2,
        )

        result = service.generate_chat_answer(
            user=user,
            prompt_build=built,
            input_risk_action=RiskAction.ALLOW,
            input_risk_level="low",
        )

        self.assertIn("无法确认", result.answer)
        self.assertEqual(result.action, RiskAction.ALLOW)

    def test_generate_chat_answer_builds_structured_fallback_with_evidence(self) -> None:
        service = GenerationService(
            prompt_template_service=_PromptServiceStub(),
            output_guard=OutputGuard(),
            llm_client=OpenAIClient(
                api_key=None,
                model="gpt-5.4-mini",
                base_url="https://api.openai.com/v1",
                timeout_seconds=30,
                max_output_tokens=256,
                temperature=0.0,
            ),
        )
        built = PromptBuildResult(
            scene="standard_qa",
            query="报销审批时限是什么？",
            template=PromptTemplate(
                id="prompt_standard_v1",
                scene="standard_qa",
                version=1,
                name="Standard QA",
                content="Use evidence only.",
                output_schema={"sections": "结论,依据,引用来源,限制说明"},
            ),
            rendered_prompt=RenderedPrompt(instructions="x", input_text="y"),
            assembled_context=AssembledContext(
                results=[object()],
                citations=[
                    Citation(index=1, doc_id="doc_1", title="报销制度", section_name="审批规则", version=1),
                ],
                fallback_evidence_lines=["[1] 报销审批时限为3个工作日。"],
                summary_lines=["[1] 报销制度 / 审批规则：报销审批时限为3个工作日。"],
                citation_lines=["[1] 报销制度 / 审批规则 / v1"],
                citation_text="[1] 报销制度",
            ),
        )
        user = UserContext(
            user_id="u1",
            tenant_id="t1",
            department_id="engineering",
            role="employee",
            clearance_level=2,
        )

        result = service.generate_chat_answer(
            user=user,
            prompt_build=built,
            input_risk_action=RiskAction.ALLOW,
            input_risk_level="low",
        )

        self.assertIn("结论：根据当前已授权资料，报销审批时限为3个工作日。", result.answer)
        self.assertIn("依据：", result.answer)
        self.assertIn("引用来源：[1] 报销制度 / 审批规则 / v1", result.answer)
        self.assertIn("限制说明：", result.answer)


if __name__ == "__main__":
    unittest.main()
