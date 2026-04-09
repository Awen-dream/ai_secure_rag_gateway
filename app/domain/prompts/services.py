"""Compatibility shim for prompt template management.

The concrete template lifecycle service now lives in `app.domain.prompts.template_service`.
This module preserves the older import path while callers migrate to the clearer naming.
"""

from app.domain.prompts.template_service import PromptTemplateService as PromptService

__all__ = ["PromptService"]
