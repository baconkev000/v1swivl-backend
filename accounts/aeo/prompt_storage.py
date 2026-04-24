"""
Normalize ``BusinessProfile.selected_aeo_prompts`` (legacy strings or rich dict rows).

Stored rows may be:
- ``str`` — generated / onboarding prompts (not custom).
- ``{"prompt": str, "is_custom": bool, ...}`` — optional ``type``, ``weight``, ``dynamic``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .aeo_prompts import AEOPromptType


def normalize_aeo_prompt_for_match(text: str) -> str:
    """
    Collapse internal whitespace for comparisons (matches how POST append stores custom rows).
    """
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def prompt_text_from_storage_row(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("prompt") or "").strip()
    return str(raw).strip()


def row_is_custom(raw: Any) -> bool:
    if isinstance(raw, dict):
        return bool(raw.get("is_custom"))
    return False


def count_custom_prompts_in_selected(selected_aeo_prompts: list | None) -> int:
    """How many stored rows are flagged ``is_custom`` (dict rows only; legacy strings are not custom)."""
    return sum(1 for raw in (selected_aeo_prompts or []) if row_is_custom(raw))


def monitored_prompt_keys_in_order(selected_aeo_prompts: list | None) -> list[str]:
    """Stable list of monitored prompt strings, deduped, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in selected_aeo_prompts or []:
        k = prompt_text_from_storage_row(raw)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def custom_prompt_flags_by_text(selected_aeo_prompts: list | None) -> dict[str, bool]:
    """Lowercase stripped prompt text -> is_custom (last occurrence wins)."""
    out: dict[str, bool] = {}
    for raw in selected_aeo_prompts or []:
        t = prompt_text_from_storage_row(raw)
        if not t:
            continue
        out[t.casefold()] = row_is_custom(raw)
    return out


def normalize_selected_aeo_prompts_payload(items: Iterable[Any]) -> list[Any]:
    """
    Validate and normalize PATCH payload for ``selected_aeo_prompts``.

    Returns a JSON-serializable list (strings and/or dicts) suitable for the DB field.
    """
    out: list[Any] = []
    seen: set[str] = set()
    if items is None:
        return []
    for i, raw in enumerate(items):
        if isinstance(raw, str):
            t = raw.strip()
            if not t:
                continue
            k = t.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            continue
        if isinstance(raw, dict):
            t = str(raw.get("prompt") or "").strip()
            if not t:
                raise ValueError(f"selected_aeo_prompts[{i}]: object needs non-empty \"prompt\".")
            k = t.casefold()
            if k in seen:
                continue
            seen.add(k)
            is_custom = bool(raw.get("is_custom"))
            ptype = str(raw.get("type") or "").strip().lower()
            if ptype not in {e.value for e in AEOPromptType}:
                ptype = AEOPromptType.TRANSACTIONAL.value
            try:
                w = float(raw.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            dyn = raw.get("dynamic")
            if dyn is None:
                dyn = True
            row: dict[str, Any] = {
                "prompt": t,
                "type": ptype,
                "weight": w,
                "dynamic": bool(dyn),
                "is_custom": is_custom,
            }
            out.append(row)
            continue
        raise ValueError(f"selected_aeo_prompts[{i}]: expected string or object, got {type(raw).__name__}.")
    return out


def plan_items_from_profile_selected(
    selected_aeo_prompts: list | None,
    *,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build Phase-1 / Phase-2 ``prompt_set`` dicts from stored profile JSON (mixed str/dict).
    """
    from .aeo_utils import AEO_ONBOARDING_PROMPT_COUNT, prompt_record

    cap = AEO_ONBOARDING_PROMPT_COUNT if max_items is None else max(1, int(max_items))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in selected_aeo_prompts or []:
        if isinstance(raw, dict):
            t = str(raw.get("prompt") or "").strip()
            if not t:
                continue
            h = t.casefold()
            if h in seen:
                continue
            seen.add(h)
            ptype = str(raw.get("type") or "").strip().lower()
            if ptype not in {e.value for e in AEOPromptType}:
                ptype = AEOPromptType.TRANSACTIONAL.value
            try:
                w = float(raw.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            dyn = raw.get("dynamic")
            if dyn is None:
                dyn = True
            rec = prompt_record(t, prompt_type=ptype, weight=w, dynamic=bool(dyn))
            if bool(raw.get("is_custom")):
                rec["is_custom"] = True
            out.append(rec)
        else:
            t = str(raw).strip()
            if not t:
                continue
            h = t.casefold()
            if h in seen:
                continue
            seen.add(h)
            out.append(
                prompt_record(
                    t,
                    prompt_type=AEOPromptType.TRANSACTIONAL,
                    weight=1.0,
                    dynamic=True,
                )
            )
        if len(out) >= cap:
            break
    return out


def plan_items_dicts_fallback_from_profile(profile: Any) -> list[dict[str, Any]]:
    """Phase-2 fallback when ``prompt_set`` is empty (matches legacy string-only behavior)."""
    return plan_items_from_profile_selected(getattr(profile, "selected_aeo_prompts", None) or [])
