from app.domain.prompts.models import PromptTemplate
from app.infrastructure.db.repositories.memory import store


class PromptService:
    def __init__(self) -> None:
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        if "standard_qa" in store.prompt_templates:
            return
        store.prompt_templates["standard_qa"] = [
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
        ]

    def list_templates(self) -> list[PromptTemplate]:
        items: list[PromptTemplate] = []
        for templates in store.prompt_templates.values():
            items.extend(templates)
        return items

    def add_template(self, template: PromptTemplate) -> PromptTemplate:
        store.prompt_templates.setdefault(template.scene, []).append(template)
        return template

    def get_template(self, scene: str) -> PromptTemplate:
        templates = [template for template in store.prompt_templates.get(scene, []) if template.enabled]
        if not templates:
            raise KeyError(scene)
        return max(templates, key=lambda item: item.version)
