import unittest

from app.application.evaluation.engines import NativeEvaluationExecutionEngine
from app.application.ingestion.engines import NativeDocumentIngestionEngine
from app.core.config import settings


class FrameworkEngineSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        from app.api.deps import (
            get_document_ingestion_engine,
            get_evaluation_execution_engine,
        )

        self.get_document_ingestion_engine = get_document_ingestion_engine
        self.get_evaluation_execution_engine = get_evaluation_execution_engine
        self.original_ingestion_engine = settings.ingestion_engine
        self.original_evaluation_engine = settings.evaluation_engine
        self.get_document_ingestion_engine.cache_clear()
        self.get_evaluation_execution_engine.cache_clear()

    def tearDown(self) -> None:
        settings.ingestion_engine = self.original_ingestion_engine
        settings.evaluation_engine = self.original_evaluation_engine
        self.get_document_ingestion_engine.cache_clear()
        self.get_evaluation_execution_engine.cache_clear()

    def test_native_engine_selection_by_default(self) -> None:
        settings.ingestion_engine = "native"
        settings.evaluation_engine = "native"

        self.assertIsInstance(self.get_document_ingestion_engine(), NativeDocumentIngestionEngine)
        self.assertIsInstance(self.get_evaluation_execution_engine(), NativeEvaluationExecutionEngine)

    def test_llamaindex_selection_uses_skeleton_adapters(self) -> None:
        from app.infrastructure.frameworks.llamaindex_eval import LlamaIndexEvaluationExecutionEngine
        from app.infrastructure.frameworks.llamaindex_ingestion import LlamaIndexDocumentIngestionEngine

        settings.ingestion_engine = "llamaindex"
        settings.evaluation_engine = "llamaindex"
        self.get_document_ingestion_engine.cache_clear()
        self.get_evaluation_execution_engine.cache_clear()

        self.assertIsInstance(self.get_document_ingestion_engine(), LlamaIndexDocumentIngestionEngine)
        self.assertIsInstance(self.get_evaluation_execution_engine(), LlamaIndexEvaluationExecutionEngine)
