from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.domain.citations.services import Citation, build_citations
from app.domain.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class AssembledContext:
    results: list[RetrievalResult]
    citations: list[Citation]
    evidence_blocks: list[str] = field(default_factory=list)
    citation_lines: list[str] = field(default_factory=list)
    fallback_evidence_lines: list[str] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    citation_text: str = ""

    @property
    def retrieved_chunks(self) -> int:
        return len(self.results)


class ContextBuilderService:
    """Assemble retrieval evidence into prompt-ready and answer-ready context artifacts."""

    def build(self, results: list[RetrievalResult], max_evidence: int = 6, max_chunks_per_document: int = 2) -> AssembledContext:
        compact_results = self._compact_results(results, max_evidence=max_evidence, max_chunks_per_document=max_chunks_per_document)
        citations = build_citations(compact_results)
        citation_by_doc_id = {item.doc_id: item for item in citations}

        evidence_blocks: list[str] = []
        fallback_evidence_lines: list[str] = []
        summary_lines: list[str] = []
        for index, result in enumerate(compact_results, start=1):
            citation = citation_by_doc_id.get(result.document.id)
            citation_label = citation.index if citation else index
            sources = ", ".join(result.retrieval_sources) or "retrieval"
            snippet = self._snippet(result.chunk.text, limit=180)
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
            fallback_evidence_lines.append(f"[{citation_label}] {snippet}")
            summary_lines.append(f"[{citation_label}] {result.document.title} / {result.chunk.section_name}：{snippet}")

        citation_lines = [
            f"[{item.index}] {item.title} / {item.section_name} / v{item.version}"
            for item in citations
        ]
        citation_text = ", ".join(f"[{item.index}] {item.title}" for item in citations)
        return AssembledContext(
            results=compact_results,
            citations=citations,
            evidence_blocks=evidence_blocks,
            citation_lines=citation_lines,
            fallback_evidence_lines=fallback_evidence_lines,
            summary_lines=summary_lines,
            citation_text=citation_text,
        )

    @staticmethod
    def _compact_results(
        results: list[RetrievalResult],
        max_evidence: int,
        max_chunks_per_document: int,
    ) -> list[RetrievalResult]:
        compact: list[RetrievalResult] = []
        doc_counts: dict[str, int] = {}
        seen_signatures: set[str] = set()
        for result in results:
            signature = ContextBuilderService._signature(result)
            if signature in seen_signatures:
                continue
            count = doc_counts.get(result.document.id, 0)
            if count >= max_chunks_per_document:
                continue
            compact.append(result)
            seen_signatures.add(signature)
            doc_counts[result.document.id] = count + 1
            if len(compact) >= max_evidence:
                break
        return compact

    @staticmethod
    def _signature(result: RetrievalResult) -> str:
        normalized_text = re.sub(r"\s+", " ", result.chunk.text.strip().lower())
        return f"{result.document.id}::{normalized_text[:120]}"

    @staticmethod
    def _snippet(text: str, limit: int = 180) -> str:
        compact = re.sub(r"\s+", " ", text.strip())
        return compact[:limit]
