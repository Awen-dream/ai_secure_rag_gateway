from __future__ import annotations

import uuid
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from app.application.ingestion.engines import NativeDocumentIngestionEngine
from app.application.ingestion.tokenization import estimate_token_count
from app.application.query.retrieval_cache import RetrievalCache
from app.core.config import settings
from app.domain.documents.models import DocumentChunk, DocumentRecord, DocumentStatus
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore


@dataclass(frozen=True)
class _LlamaIndexIngestionComponents:
    document_class: type
    sentence_splitter_class: type | None = None
    markdown_parser_class: type | None = None


class LlamaIndexDocumentIngestionEngine(NativeDocumentIngestionEngine):
    """Use LlamaIndex node parsing when available, otherwise fall back to the native ingestion engine."""

    engine_name = "llamaindex"

    def process_document(
        self,
        *,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
        retrieval_cache: RetrievalCache | None,
        doc_id: str,
    ) -> DocumentRecord:
        components = self._load_components()
        if components is None:
            return super().process_document(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                retrieval_cache=retrieval_cache,
                doc_id=doc_id,
            )

        document = repository.get_document(doc_id)
        if not document:
            raise KeyError(doc_id)

        try:
            source_bytes = source_store.read_source(document.id, document.source_type)
            self._update_document_status(repository, document, DocumentStatus.PARSING)

            content = self._extract_text(document, source_bytes)
            self._update_document_status(repository, document, DocumentStatus.CHUNKING)
            chunks = self._build_chunks_with_llamaindex(document, content, components)
        except Exception:
            return super().process_document(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                retrieval_cache=retrieval_cache,
                doc_id=doc_id,
            )

        try:
            self._update_document_status(repository, document, DocumentStatus.EMBEDDING)
            self._update_document_status(repository, document, DocumentStatus.INDEXING)

            previous_records = self._resolve_previous_records(repository, document)
            document.status = DocumentStatus.SUCCESS
            document.last_error = None
            document.current = True
            document.updated_at = self._utcnow()
            repository.save_document(document, chunks, [item.id for item in previous_records])
            self._mark_replaced_versions(repository, document, previous_records)
            indexing_service.upsert_document(document, chunks)
            if retrieval_cache:
                retrieval_cache.invalidate_all()
            return document
        except Exception:
            return super().process_document(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                retrieval_cache=retrieval_cache,
                doc_id=doc_id,
            )

    @staticmethod
    def _extract_text(document: DocumentRecord, source_bytes: bytes) -> str:
        from app.application.ingestion.document_parser import extract_text_from_bytes

        return extract_text_from_bytes(source_bytes, document.source_type, document.title)

    @staticmethod
    def _utcnow():
        from app.application.ingestion.engines import utcnow

        return utcnow()

    def _load_components(self) -> _LlamaIndexIngestionComponents | None:
        try:
            schema_module = import_module("llama_index.core.schema")
            parser_module = import_module("llama_index.core.node_parser")
        except Exception:
            return None

        document_class = getattr(schema_module, "Document", None)
        sentence_splitter_class = getattr(parser_module, "SentenceSplitter", None)
        markdown_parser_class = getattr(parser_module, "MarkdownNodeParser", None)
        if document_class is None or (sentence_splitter_class is None and markdown_parser_class is None):
            return None
        return _LlamaIndexIngestionComponents(
            document_class=document_class,
            sentence_splitter_class=sentence_splitter_class,
            markdown_parser_class=markdown_parser_class,
        )

    def _build_chunks_with_llamaindex(
        self,
        document: DocumentRecord,
        content: str,
        components: _LlamaIndexIngestionComponents,
    ) -> list[DocumentChunk]:
        li_document = components.document_class(
            text=content,
            metadata={
                "title": document.title,
                "source_type": document.source_type,
                "doc_id": document.id,
            },
        )
        nodes = self._build_nodes(li_document, document.source_type, components)
        chunks: list[DocumentChunk] = []
        for index, node in enumerate(nodes):
            text = self._extract_node_text(node).strip()
            if not text:
                continue
            metadata = dict(getattr(node, "metadata", {}) or {})
            section_name = self._resolve_section_name(metadata, document)
            heading_path = self._resolve_heading_path(metadata, section_name)
            chunks.append(
                DocumentChunk(
                    id=f"chunk_{uuid.uuid4().hex[:12]}",
                    doc_id=document.id,
                    tenant_id=document.tenant_id,
                    chunk_index=index,
                    section_name=section_name,
                    text=text,
                    token_count=estimate_token_count(text),
                    security_level=document.security_level,
                    department_scope=document.department_scope,
                    metadata_json={
                        "title": document.title,
                        "section_name": section_name,
                        "heading_path": " > ".join(heading_path),
                        "node_parser": self.engine_name,
                    },
                )
            )

        if not chunks:
            raise ValueError("LlamaIndex parser produced no chunks.")
        return chunks

    def _build_nodes(
        self,
        li_document: Any,
        source_type: str,
        components: _LlamaIndexIngestionComponents,
    ) -> list[Any]:
        normalized_type = (source_type or "").lower()
        if normalized_type == "markdown" and components.markdown_parser_class is not None:
            parser = self._instantiate_markdown_parser(components.markdown_parser_class)
            nodes = self._call_document_parser(parser, [li_document])
            if nodes:
                return nodes

        splitter_class = components.sentence_splitter_class
        if splitter_class is None:
            raise RuntimeError("SentenceSplitter is unavailable.")
        splitter = self._instantiate_sentence_splitter(splitter_class)
        nodes = self._call_document_parser(splitter, [li_document])
        if nodes:
            return nodes

        split_text = getattr(splitter, "split_text", None)
        if callable(split_text):
            texts = split_text(getattr(li_document, "text", ""))
            return [type("TextNode", (), {"text": item, "metadata": getattr(li_document, "metadata", {})}) for item in texts]
        raise RuntimeError("Unable to split document with LlamaIndex.")

    @staticmethod
    def _instantiate_sentence_splitter(splitter_class: type) -> Any:
        if hasattr(splitter_class, "from_defaults"):
            return splitter_class.from_defaults(
                chunk_size=settings.chunk_max_tokens,
                chunk_overlap=settings.chunk_overlap_tokens,
            )
        return splitter_class(
            chunk_size=settings.chunk_max_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
        )

    @staticmethod
    def _instantiate_markdown_parser(parser_class: type) -> Any:
        if hasattr(parser_class, "from_defaults"):
            return parser_class.from_defaults()
        return parser_class()

    @staticmethod
    def _call_document_parser(parser: Any, documents: list[Any]) -> list[Any]:
        for method_name in ("get_nodes_from_documents", "build_nodes_from_documents"):
            method = getattr(parser, method_name, None)
            if callable(method):
                nodes = method(documents)
                if nodes:
                    return list(nodes)
        return []

    @staticmethod
    def _extract_node_text(node: Any) -> str:
        for accessor in ("get_content", "get_text"):
            method = getattr(node, accessor, None)
            if callable(method):
                value = method()
                if isinstance(value, str):
                    return value
        return str(getattr(node, "text", "") or getattr(node, "content", "") or "")

    @staticmethod
    def _resolve_section_name(metadata: dict[str, Any], document: DocumentRecord) -> str:
        for key in ("section_name", "header", "heading", "title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return document.title

    @staticmethod
    def _resolve_heading_path(metadata: dict[str, Any], section_name: str) -> list[str]:
        for key in ("heading_path", "header_path", "headers"):
            value = metadata.get(key)
            if isinstance(value, list):
                normalized = [str(item).strip() for item in value if str(item).strip()]
                if normalized:
                    return normalized
            if isinstance(value, str) and value.strip():
                return [item.strip() for item in value.split(">") if item.strip()]
        return [section_name]
