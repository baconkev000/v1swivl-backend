"""
Google Gemini client helpers for AEO Phase 2 execution.

API key: set ``GEMINI_API_KEY`` in the environment (see Django ``GEMINI_API_KEY`` in settings).
``GOOGLE_GEMINI_API_KEY`` is accepted as a fallback when the primary var is unset.
"""

from __future__ import annotations

import logging
import os
import time

from django.conf import settings

from .models import BusinessProfile, ThirdPartyApiErrorLog, ThirdPartyApiProvider
from .third_party_usage import record_gemini_request, record_third_party_api_error

logger = logging.getLogger(__name__)


def get_effective_gemini_api_key() -> str:
    configured = (getattr(settings, "GEMINI_API_KEY", None) or "").strip()
    if configured:
        return configured
    return (os.environ.get("GOOGLE_GEMINI_API_KEY") or "").strip()


def gemini_execution_enabled() -> bool:
    return bool(get_effective_gemini_api_key())


def get_gemini_execution_model() -> str:
    return (getattr(settings, "AEO_GEMINI_EXECUTION_MODEL", None) or "gemini-2.0-flash").strip()


def _gemini_timeout_seconds() -> float:
    v = getattr(settings, "AEO_GEMINI_TIMEOUT", None)
    if v is None:
        v = getattr(settings, "AEO_OPENAI_TIMEOUT", 45.0)
    try:
        return max(5.0, float(v))
    except (TypeError, ValueError):
        return 45.0


def _gemini_max_attempts() -> int:
    v = getattr(settings, "AEO_GEMINI_MAX_ATTEMPTS", None)
    if v is None:
        v = getattr(settings, "AEO_EXECUTION_MAX_ATTEMPTS", 2)
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 2


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    try:
        from google.api_core import exceptions as google_exc

        return isinstance(
            exc,
            (
                google_exc.ResourceExhausted,
                google_exc.ServiceUnavailable,
                google_exc.DeadlineExceeded,
                google_exc.InternalServerError,
            ),
        )
    except ImportError:
        name = type(exc).__name__
        return name in ("ResourceExhausted", "ServiceUnavailable", "DeadlineExceeded", "InternalServerError")


def generate_gemini_execution_text(
    *,
    system_instruction: str,
    user_text: str,
    temperature: float,
    max_output_tokens: int,
    business_profile: BusinessProfile | None = None,
) -> tuple[str, str | None]:
    """
    Call Gemini with system + user content.

    Returns (text, error_message). error_message is set on failure.
    """
    api_key = get_effective_gemini_api_key()
    if not api_key:
        return "", "skipped_no_api_key"

    try:
        import google.generativeai as genai
    except ImportError as exc:
        logger.warning("google-generativeai not installed: %s", exc)
        return "", "gemini_sdk_missing"

    model_name = get_gemini_execution_model()
    timeout = _gemini_timeout_seconds()
    attempts = _gemini_max_attempts()
    err: str | None = None
    out = ""

    for attempt in range(attempts):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name,
                system_instruction=(system_instruction or "").strip() or None,
            )
            generation_config = genai.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )
            response = model.generate_content(
                user_text,
                generation_config=generation_config,
                request_options={"timeout": int(timeout)},
            )
            record_gemini_request(
                operation="gemini.generate_content.aeo_execution",
                response=response,
                business_profile=business_profile,
            )
            try:
                out = (response.text or "").strip()
            except (ValueError, AttributeError):
                out = ""
                if response.candidates:
                    parts = []
                    for c in response.candidates:
                        content = getattr(c, "content", None)
                        p = getattr(content, "parts", None) if content else None
                        if p:
                            for part in p:
                                t = getattr(part, "text", None)
                                if t:
                                    parts.append(str(t))
                    out = "".join(parts).strip()
            err = None
            break
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Gemini AEO execution failed (attempt %s/%s): %s",
                attempt + 1,
                attempts,
                err,
            )
            if attempt + 1 < attempts and _is_retryable_gemini_error(exc):
                time.sleep(1.0)
                continue
            break

    # One error row per logical Gemini call after retries (not per retry attempt).
    if err and err not in ("skipped_no_api_key",):
        if err == "gemini_sdk_missing":
            ek = ThirdPartyApiErrorLog.ErrorKind.VALIDATION_ERROR
        elif "DeadlineExceeded" in err or "Timeout" in err or "timeout" in err.lower():
            ek = ThirdPartyApiErrorLog.ErrorKind.TIMEOUT
        else:
            ek = ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION
        record_third_party_api_error(
            provider=ThirdPartyApiProvider.GEMINI,
            operation="gemini.generate_content.aeo_execution",
            error_kind=ek,
            message=err[:1024],
            detail=err,
            http_status=None,
            business_profile=business_profile,
        )

    return out, err
