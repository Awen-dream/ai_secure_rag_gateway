from app.domain.prompts.models import PromptTemplate
from app.infrastructure.db.repositories.sqlite import SQLiteRepository


class PromptService:
    def __init__(self, repository: SQLiteRepository) -> None:
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
                output_schema={"sections": "conclusion,evidence,citations"},
            )
        )

    def list_templates(self) -> list[PromptTemplate]:
        return self.repository.list_prompt_templates()

    def add_template(self, template: PromptTemplate) -> PromptTemplate:
        self.repository.save_prompt_template(template)
        return template

    def get_template(self, scene: str) -> PromptTemplate:
        templates = [template for template in self.repository.list_prompt_templates(scene) if template.enabled]
        if not templates:
            raise KeyError(scene)
        return max(templates, key=lambda item: item.version)
