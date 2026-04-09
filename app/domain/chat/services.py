"""Compatibility shim for chat orchestration.

The application-layer implementation now lives in `app.application.chat.orchestrator`.
This module keeps the old import path stable while the codebase migrates to the new layering.
"""

from app.application.chat.orchestrator import ChatOrchestrator as ChatService

__all__ = ["ChatService"]
