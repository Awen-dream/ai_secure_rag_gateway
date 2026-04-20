from __future__ import annotations

from importlib import import_module
from typing import Any

from app.application.ingestion.engines import NativeDocumentIngestionEngine
from app.application.query.retrieval_cache import RetrievalCache
from app.domain.documents.models import DocumentRecord
from app.domain.retrieval.indexing import RetrievalIndexingService
from app.infrastructure.db.repositories.base import MetadataRepository
from app.infrastructure.storage.local_source_store import LocalDocumentSourceStore


class LangGraphDocumentIngestionEngine(NativeDocumentIngestionEngine):
    """LangGraph-backed ingestion workflow with safe fallback to the native engine."""

    engine_name = "langgraph"

    def process_document(
        self,
        *,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
        retrieval_cache: RetrievalCache | None,
        doc_id: str,
    ) -> DocumentRecord:
        workflow = self._build_workflow(
            repository=repository,
            indexing_service=indexing_service,
            source_store=source_store,
            retrieval_cache=retrieval_cache,
        )
        if workflow is None:
            return super().process_document(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                retrieval_cache=retrieval_cache,
                doc_id=doc_id,
            )

        try:
            state = workflow.invoke({"doc_id": doc_id})
            document = state.get("document") if isinstance(state, dict) else None
            if isinstance(document, DocumentRecord):
                return document
        except Exception:
            pass
        return super().process_document(
            repository=repository,
            indexing_service=indexing_service,
            source_store=source_store,
            retrieval_cache=retrieval_cache,
            doc_id=doc_id,
        )

    def _build_workflow(
        self,
        *,
        repository: MetadataRepository,
        indexing_service: RetrievalIndexingService,
        source_store: LocalDocumentSourceStore,
        retrieval_cache: RetrievalCache | None,
    ) -> Any | None:
        try:
            graph_module = import_module("langgraph.graph")
        except Exception:
            return None

        state_graph_class = getattr(graph_module, "StateGraph", None)
        end = getattr(graph_module, "END", None)
        if state_graph_class is None or end is None:
            return None

        state_graph = state_graph_class(dict)

        def run_native(state: dict) -> dict:
            document = super(LangGraphDocumentIngestionEngine, self).process_document(
                repository=repository,
                indexing_service=indexing_service,
                source_store=source_store,
                retrieval_cache=retrieval_cache,
                doc_id=state["doc_id"],
            )
            return {"doc_id": state["doc_id"], "document": document}

        state_graph.add_node("native_ingest", run_native)
        state_graph.set_entry_point("native_ingest")
        state_graph.add_edge("native_ingest", end)
        return state_graph.compile()
