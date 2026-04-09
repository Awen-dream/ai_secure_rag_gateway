from __future__ import annotations

from dataclasses import dataclass

from app.application.context.builder import AssembledContext
from app.domain.prompts.models import PromptPreviewResponse, PromptTemplate, RenderedPrompt
from app.domain.prompts.services import PromptService


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

    def __init__(self, prompt_service: PromptService) -> None:
        self.prompt_service = prompt_service

    def build_chat_prompt(
        self,
        scene: str,
        query: str,
        assembled_context: AssembledContext,
        session_summary: str = "",
    ) -> PromptBuildResult:
        template = self.prompt_service.get_template(scene)
        rendered_prompt = self.prompt_service.render_chat_prompt(
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
