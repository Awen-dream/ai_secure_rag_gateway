from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.citations.services import Citation, build_citations
from app.domain.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class AssembledContext:
    results: list[RetrievalResult]
    citations: list[Citation]
    evidence_blocks: list[str] = field(default_factory=list)
    citation_lines: list[str] = field(default_factory=list)
    fallback_evidence_lines: list[str] = field(default_factory=list)
    citation_text: str = ""

    @property
    def retrieved_chunks(self) -> int:
        return len(self.results)


class ContextBuilderService:
    """Assemble retrieval evidence into prompt-ready and answer-ready context artifacts."""

    def build(self, results: list[RetrievalResult]) -> AssembledContext:
        citations = build_citations(results)
        citation_by_doc_id = {item.doc_id: item for item in citations}

        evidence_blocks: list[str] = []
        fallback_evidence_lines: list[str] = []
        for index, result in enumerate(results, start=1):
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
            fallback_evidence_lines.append(
                f"[{citation_label}] {result.chunk.text.replace(chr(10), ' ')[:180]}"
            )

        citation_lines = [
            f"[{item.index}] {item.title} / {item.section_name} / v{item.version}"
            for item in citations
        ]
        citation_text = ", ".join(f"[{item.index}] {item.title}" for item in citations)
        return AssembledContext(
            results=results,
            citations=citations,
            evidence_blocks=evidence_blocks,
            citation_lines=citation_lines,
            fallback_evidence_lines=fallback_evidence_lines,
            citation_text=citation_text,
        )
