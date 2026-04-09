import unittest

from app.application.context.builder import AssembledContext
from app.application.prompting.builder import PromptBuilderService
from app.domain.prompts.models import PromptTemplate, RenderedPrompt


class _PromptServiceStub:
    def get_template(self, scene: str):
        return PromptTemplate(
            id="prompt_standard_v1",
            scene=scene,
            version=1,
            name="Standard QA",
            content="Use evidence only.",
            output_schema={"sections": "结论,依据,引用来源,限制说明"},
        )

    def render_chat_prompt(self, template, query, assembled_context, session_summary=""):
        return RenderedPrompt(
            instructions=f"{template.content} session={session_summary or 'none'}",
            input_text=f"Q:{query}\nchunks={assembled_context.retrieved_chunks}",
        )


class PromptBuilderServiceTest(unittest.TestCase):
    def test_build_chat_prompt_returns_template_and_rendered_prompt(self) -> None:
        built = PromptBuilderService(_PromptServiceStub()).build_chat_prompt(
            scene="standard_qa",
            query="报销审批时限是什么？",
            assembled_context=AssembledContext(results=[], citations=[]),
            session_summary="上轮在问报销制度",
        )

        self.assertEqual(built.template.name, "Standard QA")
        self.assertIn("Use evidence only.", built.rendered_prompt.instructions)
        self.assertIn("用户问题：", built.rendered_prompt.input_text)
        self.assertIn("报销审批时限是什么？", built.rendered_prompt.input_text)


if __name__ == "__main__":
    unittest.main()
