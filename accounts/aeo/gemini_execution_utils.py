"""
AEO Phase 2 execution via Google Gemini (parallel to OpenAI in ``aeo_execution_utils``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from ..gemini_utils import gemini_execution_enabled, generate_gemini_execution_text, get_gemini_execution_model
from ..models import AEOExecutionRun, BusinessProfile
from .aeo_execution_utils import (
    PLATFORM_GEMINI,
    _execution_max_tokens,
    _execution_temperature,
    _result_dict,
    hash_prompt,
    save_aeo_response,
)
from .aeo_utils import normalize_aeo_prompt_dict
from .gemini_prompts import AEO_GEMINI_EXECUTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def run_single_aeo_prompt_gemini(
    prompt_obj: Mapping[str, Any],
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    execution_run: AEOExecutionRun | None = None,
    execution_pair_id: UUID | None = None,
) -> dict[str, Any]:
    """
    Execute one AEO prompt via Gemini; optionally save AEOResponseSnapshot (platform=gemini).

    Return shape matches ``run_single_aeo_prompt`` for batch orchestration.
    """
    spec = normalize_aeo_prompt_dict(prompt_obj)
    prompt_text = spec["prompt"]
    executed_at = datetime.now(timezone.utc)
    model = get_gemini_execution_model()

    if not gemini_execution_enabled():
        logger.info(
            "AEO Gemini execution skipped (no API key) profile_id=%s",
            getattr(business_profile, "id", None),
        )
        ph = hash_prompt(prompt_text)
        return _result_dict(
            success=False,
            spec=spec,
            raw_response="",
            model_name=model,
            platform=PLATFORM_GEMINI,
            error="skipped_no_api_key",
            snapshot_id=None,
            executed_at=executed_at,
            prompt_hash=ph,
        )

    if not prompt_text:
        return _result_dict(
            success=False,
            spec=spec,
            raw_response="",
            model_name=model,
            platform=PLATFORM_GEMINI,
            error="empty_prompt",
            snapshot_id=None,
            executed_at=executed_at,
            prompt_hash=hash_prompt(prompt_text),
        )

    ph = hash_prompt(prompt_text)
    raw, err = generate_gemini_execution_text(
        system_instruction=AEO_GEMINI_EXECUTION_SYSTEM_PROMPT,
        user_text=prompt_text,
        temperature=_execution_temperature(),
        max_output_tokens=_execution_max_tokens(),
        business_profile=business_profile,
    )

    executed_at = datetime.now(timezone.utc)
    snapshot_id: int | None = None
    success = err is None

    if success and save:
        snap = save_aeo_response(
            business_profile=business_profile,
            prompt_text=prompt_text,
            prompt_type=spec["type"],
            weight=float(spec["weight"]),
            is_dynamic=bool(spec["dynamic"]),
            raw_response=raw,
            model_name=model,
            platform=PLATFORM_GEMINI,
            prompt_hash=ph,
            execution_run=execution_run,
            execution_pair_id=execution_pair_id,
            is_custom_prompt=bool(spec.get("is_custom")),
        )
        snapshot_id = snap.id

    return _result_dict(
        success=success,
        spec=spec,
        raw_response=raw,
        model_name=model,
        platform=PLATFORM_GEMINI,
        error=err,
        snapshot_id=snapshot_id,
        executed_at=executed_at,
        prompt_hash=ph,
    )
