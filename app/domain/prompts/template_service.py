from __future__ import annotations

from app.domain.prompts.models import PromptTemplate, PromptValidationResult
from app.infrastructure.db.repositories.base import MetadataRepository


class PromptTemplateService:
    """Manage prompt templates and validate final answers against template schemas."""

    def __init__(self, repository: MetadataRepository) -> None:
        self.repository = repository
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        if self.repository.list_prompt_templates("standard_qa"):
            return
        self.repository.save_prompt_template(
            PromptTemplate(
                id="prompt_standard_v1",
                scene="standard_qa",
                version=1,
                name="Standard QA",
                content=(
                    "You are a secure enterprise knowledge assistant. "
                    "Answer only with authorized evidence. "
                    "If evidence is insufficient, say so explicitly."
                ),
                output_schema={"sections": "结论,依据,引用来源,限制说明"},
            )
        )

    def list_templates(self, scene: str | None = None) -> list[PromptTemplate]:
        """Return prompt templates, optionally filtered to one scene."""

        return self.repository.list_prompt_templates(scene)

    def add_template(self, template: PromptTemplate) -> PromptTemplate:
        """Persist one prompt template version."""

        self.repository.save_prompt_template(template)
        return template

    def set_template_enabled(self, template_id: str, enabled: bool) -> PromptTemplate:
        """Enable or disable one template by id without losing version history."""

        templates = self.repository.list_prompt_templates()
        for template in templates:
            if template.id == template_id:
                template.enabled = enabled
                self.repository.save_prompt_template(template)
                return template
        raise KeyError(template_id)

    def get_template(self, scene: str, version: int | None = None) -> PromptTemplate:
        """Return the latest enabled prompt template for one business scene."""

        templates = [template for template in self.repository.list_prompt_templates(scene) if template.enabled]
        if not templates:
            raise KeyError(scene)
        if version is not None:
            for template in templates:
                if template.version == version:
                    return template
            raise KeyError(f"{scene}:{version}")
        return max(templates, key=lambda item: item.version)

    def validate_output(self, scene: str, answer: str) -> PromptValidationResult:
        """Validate and normalize one answer against the active template output schema."""

        template = self.get_template(scene)
        required_sections = self._required_sections(template)
        missing_sections = [
            section
            for section in required_sections
            if f"{section}：" not in answer and f"{section}:" not in answer
        ]
        normalized_answer = answer.strip()
        for section in missing_sections:
            placeholder = "根据当前结果暂无补充。"
            if section == "限制说明":
                placeholder = "无。"
            normalized_answer = f"{normalized_answer}\n{section}：{placeholder}".strip()
        return PromptValidationResult(
            template_id=template.id,
            template_version=template.version,
            valid=not missing_sections,
            missing_sections=missing_sections,
            normalized_answer=normalized_answer,
        )

    @staticmethod
    def _required_sections(template: PromptTemplate) -> list[str]:
        sections = template.output_schema.get("sections", "")
        return [section.strip() for section in sections.split(",") if section.strip()]
