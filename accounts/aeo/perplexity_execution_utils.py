"""
AEO Phase 2 execution via Perplexity Sonar (OpenAI-compatible Chat Completions API).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

import requests
from django.conf import settings

from ..models import AEOExecutionRun, BusinessProfile, ThirdPartyApiErrorLog, ThirdPartyApiProvider
from ..third_party_usage import record_perplexity_request, record_third_party_api_error
from .aeo_execution_utils import (
    PLATFORM_PERPLEXITY,
    _execution_max_tokens,
    _execution_temperature,
    _result_dict,
    hash_prompt,
    save_aeo_response,
)
from .aeo_utils import normalize_aeo_prompt_dict
from .perplexity_prompts import AEO_PERPLEXITY_EXECUTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

PERPLEXITY_CHAT_COMPLETIONS_URL = "https://api.perplexity.ai/v1/chat/completions"


def perplexity_execution_enabled() -> bool:
    return bool((getattr(settings, "PERPLEXITY_API_KEY", None) or "").strip())


def get_perplexity_aeo_model() -> str:
    return (getattr(settings, "PERPLEXITY_AEO_MODEL", None) or "sonar").strip() or "sonar"


def _perplexity_timeout_seconds() -> float:
    v = getattr(settings, "PERPLEXITY_TIMEOUT", None)
    if v is None:
        v = getattr(settings, "AEO_OPENAI_TIMEOUT", 45.0)
    try:
        return max(5.0, float(v))
    except (TypeError, ValueError):
        return 45.0


def _perplexity_max_attempts() -> int:
    v = getattr(settings, "PERPLEXITY_MAX_ATTEMPTS", None)
    if v is None:
        v = getattr(settings, "AEO_EXECUTION_MAX_ATTEMPTS", 2)
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 2


def _is_retryable_perplexity_http(status_code: int) -> bool:
    return status_code in (408, 429) or (500 <= status_code <= 599)


def run_single_aeo_prompt_perplexity(
    prompt_obj: Mapping[str, Any],
    business_profile: BusinessProfile,
    *,
    save: bool = True,
    execution_run: AEOExecutionRun | None = None,
    execution_pair_id: UUID | None = None,
) -> dict[str, Any]:
    """
    Execute one AEO prompt via Perplexity Sonar; optionally save AEOResponseSnapshot (platform=perplexity).

    Return shape matches ``run_single_aeo_prompt`` / ``run_single_aeo_prompt_gemini``.
    """
    spec = normalize_aeo_prompt_dict(prompt_obj)
    prompt_text = spec["prompt"]
    executed_at = datetime.now(timezone.utc)
    model = get_perplexity_aeo_model()

    if not perplexity_execution_enabled():
        logger.info(
            "AEO Perplexity execution skipped (no API key) profile_id=%s",
            getattr(business_profile, "id", None),
        )
        ph = hash_prompt(prompt_text)
        return _result_dict(
            success=False,
            spec=spec,
            raw_response="",
            model_name=model,
            platform=PLATFORM_PERPLEXITY,
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
            platform=PLATFORM_PERPLEXITY,
            error="empty_prompt",
            snapshot_id=None,
            executed_at=executed_at,
            prompt_hash=hash_prompt(prompt_text),
        )

    ph = hash_prompt(prompt_text)
    api_key = (getattr(settings, "PERPLEXITY_API_KEY", None) or "").strip()
    timeout = _perplexity_timeout_seconds()
    attempts = _perplexity_max_attempts()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AEO_PERPLEXITY_EXECUTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": _execution_temperature(),
        "max_tokens": _execution_max_tokens(),
    }

    raw = ""
    err: str | None = None
    response_model = model
    # One error log row per logical call after retries exhaust (not per retry).
    last_http_status: int | None = None
    failure_detail: str = ""

    for attempt in range(attempts):
        try:
            resp = requests.post(
                PERPLEXITY_CHAT_COMPLETIONS_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            op = "perplexity.chat.completions"
            parsed_json = None
            if resp.content:
                try:
                    parsed_json = resp.json()
                except ValueError:
                    parsed_json = None

            usage = (parsed_json or {}).get("usage") if isinstance(parsed_json, dict) else None
            pt = ct = None
            if isinstance(usage, dict):
                try:
                    pt = int(usage.get("prompt_tokens") or 0) or None
                except (TypeError, ValueError):
                    pt = None
                try:
                    ct = int(usage.get("completion_tokens") or 0) or None
                except (TypeError, ValueError):
                    ct = None
            cost_val = None
            if isinstance(parsed_json, dict):
                c = parsed_json.get("cost")
                if c is not None:
                    try:
                        from decimal import Decimal

                        cost_val = Decimal(str(c))
                    except Exception:
                        cost_val = None

            record_perplexity_request(
                operation=f"{op} HTTP {resp.status_code}" if resp.status_code != 200 else op,
                response_status=resp.status_code,
                prompt_tokens=pt,
                completion_tokens=ct,
                cost_usd=cost_val,
                business_profile=business_profile,
            )

            last_http_status = resp.status_code
            if resp.status_code != 200:
                failure_detail = (resp.text or "")[:4000]
                err = f"HTTP {resp.status_code}: {(resp.text or '')[:500]}"
                logger.warning(
                    "AEO Perplexity execution failed (attempt %s/%s): %s",
                    attempt + 1,
                    attempts,
                    err,
                )
                if attempt + 1 < attempts and _is_retryable_perplexity_http(resp.status_code):
                    time.sleep(1.0)
                    continue
                break

            if not isinstance(parsed_json, dict):
                failure_detail = (resp.text or "")[:4000] or "non-json body"
                err = "invalid_json_response"
                break

            choices = parsed_json.get("choices") or []
            if not choices:
                failure_detail = (resp.text or "")[:4000] or "empty choices"
                err = "no_choices"
                break
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            raw = (content or "").strip() if isinstance(content, str) else ""
            rm = parsed_json.get("model")
            if isinstance(rm, str) and rm.strip():
                response_model = rm.strip()
            err = None
            break
        except requests.Timeout as exc:
            last_http_status = None
            failure_detail = str(exc)
            err = f"TimeoutError: {exc}"
            logger.warning(
                "AEO Perplexity execution failed (attempt %s/%s): %s",
                attempt + 1,
                attempts,
                err,
            )
            if attempt + 1 < attempts:
                time.sleep(1.0)
                continue
            break
        except Exception as exc:
            last_http_status = None
            failure_detail = str(exc)
            err = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "AEO Perplexity execution failed (attempt %s/%s): %s",
                attempt + 1,
                attempts,
                err,
            )
            break

    executed_at = datetime.now(timezone.utc)
    snapshot_id: int | None = None
    success = err is None

    if not success and err not in ("skipped_no_api_key", "empty_prompt"):
        if err and err.startswith("TimeoutError"):
            ek = ThirdPartyApiErrorLog.ErrorKind.TIMEOUT
        elif last_http_status is not None and last_http_status != 200:
            ek = ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR
        elif err in ("invalid_json_response", "no_choices"):
            ek = ThirdPartyApiErrorLog.ErrorKind.PARSE_ERROR
        else:
            ek = ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION
        record_third_party_api_error(
            provider=ThirdPartyApiProvider.PERPLEXITY,
            operation="perplexity.chat.completions",
            error_kind=ek,
            message=(err or "")[:1024],
            detail=failure_detail or None,
            http_status=last_http_status,
            business_profile=business_profile,
        )

    if success and save:
        snap = save_aeo_response(
            business_profile=business_profile,
            prompt_text=prompt_text,
            prompt_type=spec["type"],
            weight=float(spec["weight"]),
            is_dynamic=bool(spec["dynamic"]),
            raw_response=raw,
            model_name=response_model,
            platform=PLATFORM_PERPLEXITY,
            prompt_hash=ph,
            execution_run=execution_run,
            execution_pair_id=execution_pair_id,
        )
        snapshot_id = snap.id

    return _result_dict(
        success=success,
        spec=spec,
        raw_response=raw,
        model_name=response_model,
        platform=PLATFORM_PERPLEXITY,
        error=err,
        snapshot_id=snapshot_id,
        executed_at=executed_at,
        prompt_hash=ph,
    )
