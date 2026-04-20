from __future__ import annotations

from app.application.ingestion.engines import NativeDocumentIngestionEngine


class LlamaIndexDocumentIngestionEngine(NativeDocumentIngestionEngine):
    """Skeleton adapter for future LlamaIndex-powered ingestion while preserving native behavior today."""

    engine_name = "llamaindex"
