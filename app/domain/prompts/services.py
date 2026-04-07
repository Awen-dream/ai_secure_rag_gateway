from app.domain.citations.services import Citation
from app.domain.prompts.models import PromptTemplate, RenderedPrompt
from app.domain.retrieval.models import RetrievalResult
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
        """Return the latest enabled prompt template for one business scene."""

        templates = [template for template in self.repository.list_prompt_templates(scene) if template.enabled]
        if not templates:
            raise KeyError(scene)
        return max(templates, key=lambda item: item.version)

    def render_chat_prompt(
        self,
        template: PromptTemplate,
        query: str,
        retrieved: list[RetrievalResult],
        citations: list[Citation],
        session_summary: str = "",
    ) -> RenderedPrompt:
        """Render one authorized chat prompt from template, evidence chunks and citations."""

        citation_by_doc_id = {item.doc_id: item for item in citations}
        evidence_blocks: list[str] = []
        for index, result in enumerate(retrieved, start=1):
            citation = citation_by_doc_id.get(result.document.id)
            citation_label = citation.index if citation else index
            sources = ", ".join(result.retrieval_sources) or "retrieval"
            evidence_blocks.append(
                "\n".join(
                    [
                        f"[引用{citation_label}] 文档：{result.document.title} v{result.document.version}",
                        f"章节：{result.chunk.section_name}",
                        f"来源：{sources}",
                        f"相关度：{result.score:.4f}",
                        f"内容：{result.chunk.text.strip()}",
                    ]
                )
            )

        citation_lines = [
            f"[{item.index}] {item.title} / {item.section_name} / v{item.version}"
            for item in citations
        ]
        summary_block = session_summary.strip() or "无"
        evidence_block = "\n\n".join(evidence_blocks) if evidence_blocks else "无命中证据。"
        citation_block = "\n".join(citation_lines) if citation_lines else "无"

        instructions = "\n".join(
            [
                template.content,
                "请使用中文回答。",
                "只能基于提供的已授权资料作答，不得编造。",
                "如果证据不足，请明确回复“根据当前已授权资料无法确认”。",
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
