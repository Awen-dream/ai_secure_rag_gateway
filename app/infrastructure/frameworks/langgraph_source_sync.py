from __future__ import annotations

from importlib import import_module
from typing import Any


class LangGraphSourceSyncWorkflow:
    """LangGraph-backed source-sync workflow wrapper with safe fallback to native sync execution."""

    engine_name = "langgraph"

    def build_run_sync_job_workflow(self, run_sync_job_callable) -> Any | None:
        try:
            graph_module = import_module("langgraph.graph")
        except Exception:
            return None

        state_graph_class = getattr(graph_module, "StateGraph", None)
        end = getattr(graph_module, "END", None)
        if state_graph_class is None or end is None:
            return None

        state_graph = state_graph_class(dict)

        def execute_job(state: dict) -> dict:
            result = run_sync_job_callable(state["job_id"], state["user"])
            return {
                "job_id": state["job_id"],
                "user": state["user"],
                "summary": result,
            }

        state_graph.add_node("execute_job", execute_job)
        state_graph.set_entry_point("execute_job")
        state_graph.add_edge("execute_job", end)
        return state_graph.compile()
