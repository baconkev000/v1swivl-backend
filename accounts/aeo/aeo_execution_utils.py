"""
AEO prompt execution: OpenAI per prompt, optional parallel Google Gemini, raw response capture.

Uses existing OpenAI client resolution from accounts.openai_utils.
Execution *system* prompt text lives in aeo_prompts.py (AEO_EXECUTION_SYSTEM_PROMPT);
Gemini mirrors the same instruction via accounts.aeo.gemini_prompts.

When ``GEMINI_API_KEY`` is set, batch runs use a shared ThreadPoolExecutor (see ``AEO_EXECUTION_MAX_WORKERS``)
so many prompts’ OpenAI and Gemini calls can overlap; partial provider failure still persists the other provider’s row.

Optional Django settings (defaults are conservative for repeatable runs):
    AEO_EXECUTION_MODEL — env override; defaults to settings.OPENAI_MODEL
    AEO_EXECUTION_TEMPERATURE — default 0.2
    AEO_EXECUTION_MAX_TOKENS — default 1200 (clamped 256–4096)
    AEO_OPENAI_TIMEOUT — HTTP timeout seconds, default 45
    AEO_EXECUTION_MAX_ATTEMPTS — per-prompt retries for transient errors, default 2
    AEO_EXECUTION_MAX_WORKERS — ThreadPoolExecutor size for batch runs, default 20 (clamped 1–64)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID

from django.conf import settings

from ..models import AEOExecutionRun, AEOResponseSnapshot, BusinessProfile
from ..openai_utils import _get_client, _get_model, chat_completion_create_logged
from .aeo_prompts import AEO_EXECUTION_SYSTEM_PROMPT
from .aeo_utils import normalize_aeo_prompt_dict

logger = logging.getLogger(__name__)

PLATFORM_OPENAI = "openai"
PLATFORM_GEMINI = "gemini"
DEFAULT_PLATFORM = PLATFORM_OPENAI
DEFAULT_API_KEY_ENV = "OPEN_AI_SEO_API_KEY"


def _execution_model() -> str:
    raw = getattr(settings, "AEO_EXECUTION_MODEL", "") or ""
    s = str(raw).strip()
    return s or _get_model()


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


def _execution_max_workers() -> int:
    v = getattr(settings, "AEO_EXECUTION_MAX_WORKERS", 20)
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 20
    return max(1, min(64, n))


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
    execution_run: AEOExecutionRun | None = None,
    execution_pair_id: UUID | None = None,
) -> AEOResponseSnapshot:
    """
    Persist one execution row. Computes prompt_hash when omitted.
    """
    ph = prompt_hash or hash_prompt(prompt_text)
    return AEOResponseSnapshot.objects.create(
        profile=business_profile,
        execution_run=execution_run,
        execution_pair_id=execution_pair_id,
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
    execution_run: AEOExecutionRun | None = None,
    execution_pair_id: UUID | None = None,
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
            completion = chat_completion_create_logged(
                exec_client,
                operation="openai.chat.aeo_prompt_execution",
                business_profile=business_profile,
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
            execution_run=execution_run,
            execution_pair_id=execution_pair_id,
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


def _normalize_providers_arg(
    providers: Sequence[str] | None,
) -> tuple[str | None, str | None]:
    """
    Returns (openai_mode, gemini_mode) where each is 'only' | 'include' | None (omit).

    None / empty providers → dual-provider when Gemini key exists (auto).
    ('openai',) → OpenAI only; ('gemini',) → Gemini only.
    """
    if not providers:
        return (None, None)
    low = [str(p).strip().lower() for p in providers if str(p).strip()]
    if set(low) == {"openai"}:
        return ("only", None)
    if set(low) == {"gemini"}:
        return (None, "only")
    if set(low) >= {"openai", "gemini"}:
        return (None, None)
    return (None, None)


def run_aeo_prompt_batch(
    prompt_set: Sequence[Mapping[str, Any]],
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    platform: str = DEFAULT_PLATFORM,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    execution_run: AEOExecutionRun | None = None,
    providers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Run each prompt (OpenAI; plus Gemini in parallel when an API key is configured).

    ``providers`` — optional filter: ``['openai']`` or ``['gemini']`` runs that provider only
    (for dashboard refreshes). When omitted, uses dual execution if ``GEMINI_API_KEY`` is set.

    ``executed`` counts successful provider saves; ``failed`` counts provider attempts that
    errored (excluding Gemini skipped when no key). Celery-friendly; one provider failing
    does not block the other.

    Returns aggregate stats plus per-provider result dicts (two per prompt when dual-provider).
    """
    from ..gemini_utils import gemini_execution_enabled
    from .gemini_execution_utils import run_single_aeo_prompt_gemini

    openai_p, gemini_p = _normalize_providers_arg(providers)
    openai_only = openai_p == "only"
    gemini_only = gemini_p == "only"

    results: list[dict[str, Any]] = []
    ok = 0
    failed = 0
    dual = gemini_execution_enabled() and not openai_only and not gemini_only
    max_workers = _execution_max_workers()

    def _accumulate(one: dict[str, Any]) -> None:
        nonlocal ok, failed
        if one["success"]:
            ok += 1
        else:
            err = (one.get("error") or "").strip()
            if err != "skipped_no_api_key":
                failed += 1

    if not prompt_set:
        return {
            "profile_id": business_profile.id,
            "executed": ok,
            "failed": failed,
            "total": 0,
            "results": results,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        if gemini_only:
            futures = []
            for prompt_obj in prompt_set:
                pair_id = uuid.uuid4() if execution_run else None

                def gemini_job(
                    po: Mapping[str, Any] = prompt_obj,
                    pid: UUID | None = pair_id,
                ) -> dict[str, Any]:
                    return run_single_aeo_prompt_gemini(
                        po,
                        business_profile,
                        save=save,
                        execution_run=execution_run,
                        execution_pair_id=pid,
                    )

                futures.append(pool.submit(gemini_job))
            for fut in futures:
                one = fut.result()
                results.append(one)
                _accumulate(one)
        elif dual:
            pair_futures: list[tuple[Any, Any]] = []
            for prompt_obj in prompt_set:
                pair_id = uuid.uuid4() if execution_run else None

                def openai_job(
                    po: Mapping[str, Any] = prompt_obj,
                    pid: UUID | None = pair_id,
                ) -> dict[str, Any]:
                    return run_single_aeo_prompt(
                        po,
                        business_profile,
                        client=None,
                        save=save,
                        platform=PLATFORM_OPENAI,
                        api_key_env=api_key_env,
                        execution_run=execution_run,
                        execution_pair_id=pid,
                    )

                def gemini_job(
                    po: Mapping[str, Any] = prompt_obj,
                    pid: UUID | None = pair_id,
                ) -> dict[str, Any]:
                    return run_single_aeo_prompt_gemini(
                        po,
                        business_profile,
                        save=save,
                        execution_run=execution_run,
                        execution_pair_id=pid,
                    )

                pair_futures.append((pool.submit(openai_job), pool.submit(gemini_job)))
            for fut_o, fut_g in pair_futures:
                for one in (fut_o.result(), fut_g.result()):
                    results.append(one)
                    _accumulate(one)
        else:
            futures = []
            for prompt_obj in prompt_set:
                pair_id = uuid.uuid4() if execution_run else None

                def openai_only_job(
                    po: Mapping[str, Any] = prompt_obj,
                    pid: UUID | None = pair_id,
                ) -> dict[str, Any]:
                    return run_single_aeo_prompt(
                        po,
                        business_profile,
                        client=None,
                        save=save,
                        platform=PLATFORM_OPENAI if openai_only else platform,
                        api_key_env=api_key_env,
                        execution_run=execution_run,
                        execution_pair_id=pid,
                    )

                futures.append(pool.submit(openai_only_job))
            for fut in futures:
                one = fut.result()
                results.append(one)
                _accumulate(one)

    return {
        "profile_id": business_profile.id,
        "executed": ok,
        "failed": failed,
        "total": len(results),
        "results": results,
    }
