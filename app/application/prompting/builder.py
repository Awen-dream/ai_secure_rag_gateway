from __future__ import annotations

from dataclasses import dataclass

from app.application.context.builder import AssembledContext
from app.domain.prompts.models import PromptPreviewResponse, PromptTemplate, RenderedPrompt
from app.domain.prompts.template_service import PromptTemplateService


@dataclass(frozen=True)
class PromptBuildResult:
    scene: str
    query: str
    template: PromptTemplate
    rendered_prompt: RenderedPrompt
    assembled_context: AssembledContext
    session_summary: str = ""


class PromptBuilderService:
    """Build prompt payloads from templates and assembled retrieval context."""

    def __init__(self, prompt_template_service: PromptTemplateService) -> None:
        self.prompt_template_service = prompt_template_service

    def build_chat_prompt(
        self,
        scene: str,
        query: str,
        assembled_context: AssembledContext,
        session_summary: str = "",
    ) -> PromptBuildResult:
        template = self.prompt_template_service.get_template(scene)
        rendered_prompt = self._render_chat_prompt(
            template=template,
            query=query,
            assembled_context=assembled_context,
            session_summary=session_summary,
        )
        return PromptBuildResult(
            scene=scene,
            query=query,
            template=template,
            rendered_prompt=rendered_prompt,
            assembled_context=assembled_context,
            session_summary=session_summary,
        )

    def preview_chat_prompt(
        self,
        scene: str,
        query: str,
        assembled_context: AssembledContext,
        session_summary: str = "",
    ) -> PromptPreviewResponse:
        built = self.build_chat_prompt(
            scene=scene,
            query=query,
            assembled_context=assembled_context,
            session_summary=session_summary,
        )
        return PromptPreviewResponse(
            scene=scene,
            template_id=built.template.id,
            template_version=built.template.version,
            instructions=built.rendered_prompt.instructions,
            input_text=built.rendered_prompt.input_text,
            retrieved_chunks=assembled_context.retrieved_chunks,
        )

    @staticmethod
    def _render_chat_prompt(
        template: PromptTemplate,
        query: str,
        assembled_context: AssembledContext,
        session_summary: str = "",
    ) -> RenderedPrompt:
        summary_block = session_summary.strip() or "无"
        evidence_block = "\n\n".join(assembled_context.evidence_blocks) if assembled_context.evidence_blocks else "无命中证据。"
        citation_block = "\n".join(assembled_context.citation_lines) if assembled_context.citation_lines else "无"

        instructions = "\n".join(
            [
                template.content,
                "请使用中文回答。",
                "只能基于提供的已授权资料作答，不得编造。",
                "如果证据不足，请明确回复“根据当前已授权资料无法确认”。",
                "在依据和引用来源中使用 [1]、[2] 这样的引用编号对应证据。",
                "输出顺序固定为：结论、依据、引用来源、限制说明。",
            ]
        )
        input_text = "\n\n".join(
            [
                f"用户问题：\n{query}",
                f"会话摘要：\n{summary_block}",
                f"已授权证据：\n{evidence_block}",
                f"可用引用：\n{citation_block}",
            ]
        )
        return RenderedPrompt(instructions=instructions, input_text=input_text)
