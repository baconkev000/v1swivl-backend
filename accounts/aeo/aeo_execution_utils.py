"""
AEO prompt execution: one OpenAI call per prompt, raw response capture, DB storage.

Uses existing OpenAI client resolution from accounts.openai_utils.
Execution *system* prompt text lives in aeo_prompts.py (AEO_EXECUTION_SYSTEM_PROMPT).

Optional Django settings (defaults are conservative for repeatable runs):
    AEO_EXECUTION_MODEL — override model id (else settings.OPENAI_MODEL / gpt-4o-mini)
    AEO_EXECUTION_TEMPERATURE — default 0.2
    AEO_EXECUTION_MAX_TOKENS — default 1200 (clamped 256–4096)
    AEO_OPENAI_TIMEOUT — HTTP timeout seconds, default 45
    AEO_EXECUTION_MAX_ATTEMPTS — per-prompt retries for transient errors, default 2
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from django.conf import settings

from ..models import AEOResponseSnapshot, BusinessProfile
from ..openai_utils import _get_client, _get_model
from .aeo_prompts import AEO_EXECUTION_SYSTEM_PROMPT
from .aeo_utils import normalize_aeo_prompt_dict

logger = logging.getLogger(__name__)

DEFAULT_PLATFORM = "openai"
DEFAULT_API_KEY_ENV = "OPEN_AI_SEO_API_KEY"


def _execution_model() -> str:
    return getattr(settings, "AEO_EXECUTION_MODEL", None) or _get_model()


def _execution_temperature() -> float:
    v = getattr(settings, "AEO_EXECUTION_TEMPERATURE", 0.2)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.2


def _execution_max_tokens() -> int:
    v = getattr(settings, "AEO_EXECUTION_MAX_TOKENS", 1200)
    try:
        return max(256, min(4096, int(v)))
    except (TypeError, ValueError):
        return 1200


def _execution_timeout_seconds() -> float:
    v = getattr(settings, "AEO_OPENAI_TIMEOUT", 45.0)
    try:
        return max(5.0, float(v))
    except (TypeError, ValueError):
        return 45.0


def _execution_max_attempts() -> int:
    v = getattr(settings, "AEO_EXECUTION_MAX_ATTEMPTS", 2)
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 2


def normalize_prompt_for_hash(prompt_text: str) -> str:
    """
    Canonical form for stable hashing (same consumer intent → same hash).
    """
    s = unicodedata.normalize("NFC", (prompt_text or "").strip())
    return re.sub(r"\s+", " ", s).strip()


def hash_prompt(prompt_text: str) -> str:
    """
    Stable SHA-256 hex digest of the normalized prompt text.
    """
    normalized = normalize_prompt_for_hash(prompt_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _light_trim_response(text: str | None) -> str:
    """Trim leading/trailing whitespace only; preserve inner content."""
    if text is None:
        return ""
    return text.strip()


def _execution_openai_client(api_key_env: str):
    """
    Client with bounded timeout and no SDK-level retries (we retry per prompt).
    """
    base = _get_client(api_key_env)
    timeout = _execution_timeout_seconds()
    try:
        return base.with_options(timeout=timeout, max_retries=0)
    except Exception:
        return base


def _is_retryable_openai_error(exc: BaseException) -> bool:
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        return isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError))
    except ImportError:
        name = type(exc).__name__
        return name in ("APITimeoutError", "APIConnectionError", "RateLimitError")


def save_aeo_response(
    *,
    business_profile: BusinessProfile,
    prompt_text: str,
    prompt_type: str,
    weight: float,
    is_dynamic: bool,
    raw_response: str,
    model_name: str,
    platform: str = DEFAULT_PLATFORM,
    prompt_hash: str | None = None,
) -> AEOResponseSnapshot:
    """
    Persist one execution row. Computes prompt_hash when omitted.
    """
    ph = prompt_hash or hash_prompt(prompt_text)
    return AEOResponseSnapshot.objects.create(
        profile=business_profile,
        prompt_text=prompt_text,
        prompt_type=(prompt_type or "")[:32],
        weight=float(weight),
        is_dynamic=bool(is_dynamic),
        platform=(platform or DEFAULT_PLATFORM)[:64],
        model_name=(model_name or "")[:128],
        raw_response=raw_response,
        prompt_hash=ph,
    )


def run_single_aeo_prompt(
    prompt_obj: Mapping[str, Any],
    business_profile: BusinessProfile,
    *,
    client: Any | None = None,
    save: bool = True,
    platform: str = DEFAULT_PLATFORM,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, Any]:
    """
    Execute one AEO prompt via OpenAI; optionally save AEOResponseSnapshot.

    prompt_obj: mapping with prompt, type, weight, dynamic (Phase 1 shape).
    """
    spec = normalize_aeo_prompt_dict(prompt_obj)
    prompt_text = spec["prompt"]
    executed_at = datetime.now(timezone.utc)
    if not prompt_text:
        return _result_dict(
            success=False,
            spec=spec,
            raw_response="",
            model_name=_execution_model(),
            platform=platform,
            error="empty_prompt",
            snapshot_id=None,
            executed_at=executed_at,
            prompt_hash=hash_prompt(prompt_text),
        )

    ph = hash_prompt(prompt_text)
    model = _execution_model()
    owned_client = client is None
    exec_client = client or _execution_openai_client(api_key_env)

    logger.debug(
        "AEO execute prompt for profile_id=%s",
        getattr(business_profile, "id", None),
    )

    raw = ""
    err: str | None = None
    attempts = _execution_max_attempts()
    for attempt in range(attempts):
        try:
            completion = exec_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": AEO_EXECUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=_execution_temperature(),
                max_tokens=_execution_max_tokens(),
            )
            raw = _light_trim_response(
                (completion.choices[0].message.content or "")
                if completion.choices
                else ""
            )
            err = None
            break
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "AEO OpenAI execution failed (attempt %s/%s): %s",
                attempt + 1,
                attempts,
                err,
            )
            if attempt + 1 < attempts and _is_retryable_openai_error(exc):
                time.sleep(1.0)
                continue
            break

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
            platform=platform,
            prompt_hash=ph,
        )
        snapshot_id = snap.id

    if owned_client and hasattr(exec_client, "close"):
        try:
            exec_client.close()
        except Exception:
            pass

    return _result_dict(
        success=success,
        spec=spec,
        raw_response=raw,
        model_name=model,
        platform=platform,
        error=err,
        snapshot_id=snapshot_id,
        executed_at=executed_at,
        prompt_hash=ph,
    )


def _result_dict(
    *,
    success: bool,
    spec: dict[str, Any],
    raw_response: str,
    model_name: str,
    platform: str,
    error: str | None,
    snapshot_id: int | None,
    executed_at: datetime | None,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "success": success,
        "prompt": spec.get("prompt", ""),
        "type": spec.get("type", ""),
        "weight": spec.get("weight", 1.0),
        "dynamic": spec.get("dynamic", False),
        "raw_response": raw_response,
        "model_name": model_name,
        "platform": platform,
        "error": error,
        "snapshot_id": snapshot_id,
        "executed_at": executed_at.isoformat() if executed_at else None,
        "prompt_hash": prompt_hash,
    }


def run_aeo_prompt_batch(
    prompt_set: Sequence[Mapping[str, Any]],
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    platform: str = DEFAULT_PLATFORM,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, Any]:
    """
    Run each prompt independently (Celery-friendly). Failures do not stop the batch.

    Returns aggregate stats plus per-prompt result dicts.
    """
    exec_client = _execution_openai_client(api_key_env)
    try:
        results: list[dict[str, Any]] = []
        ok = 0
        failed = 0
        for prompt_obj in prompt_set:
            one = run_single_aeo_prompt(
                prompt_obj,
                business_profile,
                client=exec_client,
                save=save,
                platform=platform,
                api_key_env=api_key_env,
            )
            results.append(one)
            if one["success"]:
                ok += 1
            else:
                failed += 1
        return {
            "profile_id": business_profile.id,
            "executed": ok,
            "failed": failed,
            "total": len(results),
            "results": results,
        }
    finally:
        if hasattr(exec_client, "close"):
            try:
                exec_client.close()
            except Exception:
                pass
