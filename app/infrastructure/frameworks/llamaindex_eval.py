from __future__ import annotations

from app.application.evaluation.engines import NativeEvaluationExecutionEngine


class LlamaIndexEvaluationExecutionEngine(NativeEvaluationExecutionEngine):
    """Skeleton adapter for future LlamaIndex-backed evaluation while preserving native behavior today."""

    engine_name = "llamaindex"
