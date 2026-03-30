"""
Gemini-specific AEO Phase 2 (execution) prompt constants.

Content matches OpenAI execution behavior; defined here so Gemini strings stay out of
generic execution modules.
"""

from __future__ import annotations

from typing import Final

from .aeo_prompts import AEO_EXECUTION_SYSTEM_PROMPT

AEO_GEMINI_EXECUTION_SYSTEM_PROMPT: Final[str] = AEO_EXECUTION_SYSTEM_PROMPT
