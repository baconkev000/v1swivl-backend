"""
Record and aggregate third-party API usage (DataForSEO, OpenAI, Gemini, Perplexity) for staff dashboards.
Failed calls can be persisted separately via ``record_third_party_api_error`` (debugging; one row per logical
failure—call from the outermost layer when retries are internal, not once per retry).

DataForSEO: each HTTP call is billed per API task; ``cost_usd`` is the sum of ``tasks[].cost`` from the
response body when present (not token-based). OpenAI/Gemini rows use token metadata with USD estimates.
Perplexity rows log reported token counts; ``cost_usd`` is set only when the API returns an explicit cost field.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from decimal import Decimal
from typing import Any, Generator

from django.conf import settings
from django.db.models import Count, IntegerField, Sum, Value
from django.db.models.functions import Coalesce, TruncMonth
from django.utils import timezone

from accounts.aeo.aeo_scoring_utils import (
    calculate_visibility_score,
    latest_extraction_per_response_in_window,
)
from accounts.models import (
    AEOPromptExecutionAggregate,
    AEOResponseSnapshot,
    BusinessProfile,
    ThirdPartyApiErrorLog,
    ThirdPartyApiProvider,
    ThirdPartyApiRequestLog,
)

logger = logging.getLogger(__name__)

# Max length stored on ThirdPartyApiErrorLog.detail (avoid huge bodies; no full prompts).
THIRD_PARTY_ERROR_DETAIL_MAX_CHARS = 6000

_usage_business_profile: ContextVar[BusinessProfile | None] = ContextVar(
    "usage_business_profile",
    default=None,
)


@contextmanager
def usage_profile_context(
    business_profile: BusinessProfile | None,
) -> Generator[None, None, None]:
    """Set the active BusinessProfile for third-party usage attribution (thread/async safe)."""
    token = _usage_business_profile.set(business_profile)
    try:
        yield
    finally:
        _usage_business_profile.reset(token)


def get_usage_business_profile() -> BusinessProfile | None:
    return _usage_business_profile.get()


def effective_usage_profile(explicit: BusinessProfile | None) -> BusinessProfile | None:
    if explicit is not None:
        return explicit
    return get_usage_business_profile()


def _truncate_third_party_error_detail(text: str | None) -> str:
    s = (text or "").strip()
    if len(s) <= THIRD_PARTY_ERROR_DETAIL_MAX_CHARS:
        return s
    return s[:THIRD_PARTY_ERROR_DETAIL_MAX_CHARS] + "\n…[truncated]"


def classify_openai_sdk_exception(exc: BaseException) -> tuple[str, int | None]:
    """
    Map an OpenAI SDK (or generic) exception to (ThirdPartyApiErrorLog.ErrorKind value, http_status_or_none).
    """
    http_raw = getattr(exc, "status_code", None)
    if http_raw is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            http_raw = getattr(resp, "status_code", None)
    http: int | None = None
    if http_raw is not None:
        try:
            http = int(http_raw)
        except (TypeError, ValueError):
            http = None

    try:
        from openai import APIConnectionError, APITimeoutError
        from openai import APIStatusError

        if isinstance(exc, APITimeoutError):
            return ThirdPartyApiErrorLog.ErrorKind.TIMEOUT, http
        if isinstance(exc, APIConnectionError):
            return ThirdPartyApiErrorLog.ErrorKind.CONNECTION_ERROR, http
        if isinstance(exc, APIStatusError):
            return ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR, http
    except ImportError:
        pass

    name = type(exc).__name__
    if "Timeout" in name:
        return ThirdPartyApiErrorLog.ErrorKind.TIMEOUT, http
    if "Connection" in name or "ConnectError" in name:
        return ThirdPartyApiErrorLog.ErrorKind.CONNECTION_ERROR, http
    if http is not None:
        return ThirdPartyApiErrorLog.ErrorKind.HTTP_ERROR, http
    return ThirdPartyApiErrorLog.ErrorKind.UNKNOWN_EXCEPTION, http


def record_third_party_api_error(
    *,
    provider: str,
    operation: str,
    error_kind: str,
    message: str,
    detail: str | None = None,
    http_status: int | None = None,
    business_profile: BusinessProfile | None = None,
) -> None:
    """
    Persist one outbound API failure row. Never raises (logs and swallows DB errors).
    """
    try:
        ThirdPartyApiErrorLog.objects.create(
            provider=provider,
            business_profile=effective_usage_profile(business_profile),
            operation=(operation or "")[:512],
            http_status=http_status,
            error_kind=error_kind,
            message=(message or "")[:1024],
            detail=_truncate_third_party_error_detail(detail),
        )
    except Exception:
        logger.exception("record_third_party_api_error failed op=%s provider=%s", operation, provider)


# Rough USD per 1M tokens when usage is not itemized (override via settings)
_DEFAULT_OPENAI_INPUT_PER_M = Decimal("2.50")
_DEFAULT_OPENAI_OUTPUT_PER_M = Decimal("10.00")


def _openai_price_per_m(model: str | None) -> tuple[Decimal, Decimal]:
    pricing = getattr(settings, "OPENAI_USAGE_PRICING_PER_MILLION_TOKENS", None) or {}
    if model and isinstance(pricing, dict) and model in pricing:
        p = pricing[model]
        inp = Decimal(str(p.get("input", _DEFAULT_OPENAI_INPUT_PER_M)))
        out = Decimal(str(p.get("output", _DEFAULT_OPENAI_OUTPUT_PER_M)))
        return inp, out
    return _DEFAULT_OPENAI_INPUT_PER_M, _DEFAULT_OPENAI_OUTPUT_PER_M


def record_dataforseo_request(
    *,
    operation: str,
    response_json: dict[str, Any] | None,
    business_profile: BusinessProfile | None = None,
) -> None:
    try:
        cost: Decimal | None = None
        if response_json is not None:
            tasks = response_json.get("tasks") or []
            total = Decimal("0")
            any_cost = False
            if isinstance(tasks, list):
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    c = task.get("cost")
                    if c is None:
                        continue
                    try:
                        total += Decimal(str(c))
                        any_cost = True
                    except Exception:
                        continue
            if any_cost:
                cost = total
        bp = effective_usage_profile(business_profile)
        ThirdPartyApiRequestLog.objects.create(
            provider=ThirdPartyApiRequestLog.Provider.DATAFORSEO,
            business_profile=bp,
            operation=(operation or "")[:512],
            cost_usd=cost,
        )
    except Exception:
        logger.exception("record_dataforseo_request failed op=%s", operation)


def record_openai_chat_completion(
    *,
    operation: str,
    response: Any,
    business_profile: BusinessProfile | None = None,
) -> None:
    try:
        cost = None
        model = None
        tokens_sent: int | None = None
        tokens_received: int | None = None
        if response is not None:
            model = getattr(response, "model", None)
            usage = getattr(response, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                if pt is not None:
                    try:
                        tokens_sent = max(0, int(pt))
                    except (TypeError, ValueError):
                        tokens_sent = None
                if ct is not None:
                    try:
                        tokens_received = max(0, int(ct))
                    except (TypeError, ValueError):
                        tokens_received = None
                pt_d = int(pt or 0)
                ct_d = int(ct or 0)
                inp_m, out_m = _openai_price_per_m(model)
                cost = (Decimal(pt_d) / Decimal(1_000_000)) * inp_m + (
                    Decimal(ct_d) / Decimal(1_000_000)
                ) * out_m
        ThirdPartyApiRequestLog.objects.create(
            provider=ThirdPartyApiRequestLog.Provider.OPENAI,
            business_profile=effective_usage_profile(business_profile),
            operation=(operation or "")[:512],
            cost_usd=cost,
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
        )
    except Exception:
        logger.exception("record_openai_chat_completion failed op=%s", operation)


def _gemini_price_per_m() -> tuple[Decimal, Decimal]:
    return (
        Decimal(str(getattr(settings, "GEMINI_INPUT_USD_PER_MILLION", "0.075"))),
        Decimal(str(getattr(settings, "GEMINI_OUTPUT_USD_PER_MILLION", "0.30"))),
    )


def record_perplexity_request(
    *,
    operation: str,
    response_status: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cost_usd: Decimal | None = None,
    business_profile: BusinessProfile | None = None,
) -> None:
    """
    Log a Perplexity Sonar Chat Completions call. No USD estimate from tokens; ``cost_usd`` only if
    the API returns a cost field. Token counts stored when ``usage`` is present for staff charts.
    """
    try:
        ThirdPartyApiRequestLog.objects.create(
            provider=ThirdPartyApiRequestLog.Provider.PERPLEXITY,
            business_profile=effective_usage_profile(business_profile),
            operation=(operation or "")[:512],
            cost_usd=cost_usd,
            tokens_sent=max(0, int(prompt_tokens)) if prompt_tokens is not None else None,
            tokens_received=max(0, int(completion_tokens)) if completion_tokens is not None else None,
        )
    except Exception:
        logger.exception("record_perplexity_request failed op=%s", operation)


def record_gemini_request(
    *,
    operation: str,
    response: Any,
    business_profile: BusinessProfile | None = None,
) -> None:
    try:
        cost = None
        tokens_sent: int | None = None
        tokens_received: int | None = None
        if response is not None:
            um = getattr(response, "usage_metadata", None)
            if um is not None:
                pt = getattr(um, "prompt_token_count", None)
                ct = getattr(um, "candidates_token_count", None)
                if pt is not None:
                    try:
                        tokens_sent = max(0, int(pt))
                    except (TypeError, ValueError):
                        tokens_sent = None
                if ct is not None:
                    try:
                        tokens_received = max(0, int(ct))
                    except (TypeError, ValueError):
                        tokens_received = None
                pt_d = int(pt or 0)
                ct_d = int(ct or 0)
                inp_m, out_m = _gemini_price_per_m()
                cost = (Decimal(pt_d) / Decimal(1_000_000)) * inp_m + (
                    Decimal(ct_d) / Decimal(1_000_000)
                ) * out_m
        ThirdPartyApiRequestLog.objects.create(
            provider=ThirdPartyApiRequestLog.Provider.GEMINI,
            business_profile=effective_usage_profile(business_profile),
            operation=(operation or "")[:512],
            cost_usd=cost,
            tokens_sent=tokens_sent,
            tokens_received=tokens_received,
        )
    except Exception:
        logger.exception("record_gemini_request failed op=%s", operation)


def _last_n_calendar_month_starts(n: int) -> list[tuple[int, int]]:
    ref = timezone.localdate()
    y, m = ref.year, ref.month
    seq: list[tuple[int, int]] = []
    for _ in range(n):
        seq.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    seq.reverse()
    return seq


def _month_label(y: int, mo: int) -> str:
    return datetime(y, mo, 1).strftime("%b %Y")


def build_monthly_api_usage_chart_context(
    business_profile_id: int | None,
    *,
    months: int = 12,
) -> dict[str, Any]:
    """
    Context for staff home chart: Chart.js datasets + labels, plus totals.
    business_profile_id None = all profiles.
    """
    month_keys = _last_n_calendar_month_starts(months)
    labels = [_month_label(y, mo) for y, mo in month_keys]
    first_y, first_m = month_keys[0]
    start_naive = datetime(first_y, first_m, 1, 0, 0, 0)
    if timezone.is_naive(start_naive):
        start_dt = timezone.make_aware(start_naive, timezone.get_current_timezone())
    else:
        start_dt = start_naive

    qs = ThirdPartyApiRequestLog.objects.filter(created_at__gte=start_dt)
    if business_profile_id is not None:
        qs = qs.filter(business_profile_id=business_profile_id)

    rows = (
        qs.annotate(m=TruncMonth("created_at", tzinfo=timezone.get_current_timezone()))
        .values("m", "provider")
        .annotate(cnt=Count("id"), cost=Sum("cost_usd"))
    )

    # (year, month) -> provider -> {count, cost}
    by_month_provider: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "cost": Decimal("0")})
    )
    for row in rows:
        dt = row["m"]
        if dt is None:
            continue
        key = (dt.year, dt.month)
        prov = row["provider"]
        by_month_provider[key][prov]["count"] = row["cnt"]
        by_month_provider[key][prov]["cost"] = row["cost"] or Decimal("0")

    providers = [
        ThirdPartyApiRequestLog.Provider.DATAFORSEO,
        ThirdPartyApiRequestLog.Provider.OPENAI,
        ThirdPartyApiRequestLog.Provider.GEMINI,
        ThirdPartyApiRequestLog.Provider.PERPLEXITY,
    ]
    provider_labels = {
        ThirdPartyApiRequestLog.Provider.DATAFORSEO: "DataForSEO",
        ThirdPartyApiRequestLog.Provider.OPENAI: "OpenAI",
        ThirdPartyApiRequestLog.Provider.GEMINI: "Gemini",
        ThirdPartyApiRequestLog.Provider.PERPLEXITY: "Perplexity",
    }
    colors = {
        ThirdPartyApiRequestLog.Provider.DATAFORSEO: "rgba(54, 162, 235, 0.7)",
        ThirdPartyApiRequestLog.Provider.OPENAI: "rgba(75, 192, 192, 0.7)",
        ThirdPartyApiRequestLog.Provider.GEMINI: "rgba(255, 159, 64, 0.7)",
        ThirdPartyApiRequestLog.Provider.PERPLEXITY: "rgba(124, 58, 237, 0.7)",
    }

    count_datasets = []
    cost_datasets = []
    for p in providers:
        count_data = []
        cost_data = []
        for y, mo in month_keys:
            cell = by_month_provider[(y, mo)].get(p, {"count": 0, "cost": Decimal("0")})
            count_data.append(cell["count"])
            cost_data.append(float(cell["cost"]))
        count_datasets.append(
            {
                "label": provider_labels[p],
                "data": count_data,
                "backgroundColor": colors[p],
            }
        )
        cost_datasets.append(
            {
                "label": provider_labels[p],
                "data": cost_data,
                "backgroundColor": colors[p],
            }
        )

    total_requests = sum(
        by_month_provider[k][p]["count"] for k in month_keys for p in providers
    )
    total_cost = sum(
        (by_month_provider[k][p]["cost"] or Decimal("0"))
        for k in month_keys
        for p in providers
    )

    return {
        "api_usage_chart_labels": labels,
        "api_usage_count_datasets": count_datasets,
        "api_usage_cost_datasets": cost_datasets,
        "api_usage_total_requests": total_requests,
        "api_usage_total_cost_usd": float(total_cost),
        "api_usage_profile_filter": business_profile_id,
    }


def build_monthly_token_usage_chart_context(
    business_profile_id: int | None,
    tokens_platform: str,
    *,
    months: int = 12,
) -> dict[str, Any]:
    """
    Monthly sums of tokens_sent / tokens_received for staff charts.
    ``tokens_platform``: ``all`` | ``dataforseo`` | ``openai`` | ``gemini`` | ``perplexity``.
    """
    month_keys = _last_n_calendar_month_starts(months)
    labels = [_month_label(y, mo) for y, mo in month_keys]
    first_y, first_m = month_keys[0]
    start_naive = datetime(first_y, first_m, 1, 0, 0, 0)
    if timezone.is_naive(start_naive):
        start_dt = timezone.make_aware(start_naive, timezone.get_current_timezone())
    else:
        start_dt = start_naive

    qs = ThirdPartyApiRequestLog.objects.filter(created_at__gte=start_dt)
    if business_profile_id is not None:
        qs = qs.filter(business_profile_id=business_profile_id)

    providers = [
        ThirdPartyApiRequestLog.Provider.DATAFORSEO,
        ThirdPartyApiRequestLog.Provider.OPENAI,
        ThirdPartyApiRequestLog.Provider.GEMINI,
        ThirdPartyApiRequestLog.Provider.PERPLEXITY,
    ]
    provider_labels = {
        ThirdPartyApiRequestLog.Provider.DATAFORSEO: "DataForSEO",
        ThirdPartyApiRequestLog.Provider.OPENAI: "OpenAI",
        ThirdPartyApiRequestLog.Provider.GEMINI: "Gemini",
        ThirdPartyApiRequestLog.Provider.PERPLEXITY: "Perplexity",
    }
    colors = {
        ThirdPartyApiRequestLog.Provider.DATAFORSEO: "rgba(54, 162, 235, 0.75)",
        ThirdPartyApiRequestLog.Provider.OPENAI: "rgba(75, 192, 192, 0.75)",
        ThirdPartyApiRequestLog.Provider.GEMINI: "rgba(255, 159, 64, 0.75)",
        ThirdPartyApiRequestLog.Provider.PERPLEXITY: "rgba(124, 58, 237, 0.75)",
    }

    tp = (tokens_platform or "all").strip().lower()
    if tp not in ("all", "dataforseo", "openai", "gemini", "perplexity"):
        tp = "all"

    tz = timezone.get_current_timezone()
    zero = Value(0, output_field=IntegerField())

    if tp == "all":
        rows = (
            qs.annotate(m=TruncMonth("created_at", tzinfo=tz))
            .values("m", "provider")
            .annotate(
                sent=Coalesce(Sum("tokens_sent"), zero),
                recv=Coalesce(Sum("tokens_received"), zero),
            )
        )
        by_month_provider: dict[tuple[int, int], dict[str, dict[str, int]]] = defaultdict(dict)
        for row in rows:
            dt = row["m"]
            if dt is None:
                continue
            key = (dt.year, dt.month)
            prov = row["provider"]
            by_month_provider[key][prov] = {
                "sent": int(row["sent"] or 0),
                "recv": int(row["recv"] or 0),
            }
        sent_datasets: list[dict[str, Any]] = []
        recv_datasets: list[dict[str, Any]] = []
        for p in providers:
            sent_data: list[int] = []
            recv_data: list[int] = []
            for y, mo in month_keys:
                cell = by_month_provider[(y, mo)].get(p, {"sent": 0, "recv": 0})
                sent_data.append(cell["sent"])
                recv_data.append(cell["recv"])
            sent_datasets.append(
                {
                    "label": provider_labels[p],
                    "data": sent_data,
                    "backgroundColor": colors[p],
                }
            )
            recv_datasets.append(
                {
                    "label": provider_labels[p],
                    "data": recv_data,
                    "backgroundColor": colors[p],
                }
            )
    else:
        qs = qs.filter(provider=tp)
        rows = (
            qs.annotate(m=TruncMonth("created_at", tzinfo=tz))
            .values("m")
            .annotate(
                sent=Coalesce(Sum("tokens_sent"), zero),
                recv=Coalesce(Sum("tokens_received"), zero),
            )
        )
        by_month: dict[tuple[int, int], dict[str, int]] = {}
        for row in rows:
            dt = row["m"]
            if dt is None:
                continue
            by_month[(dt.year, dt.month)] = {
                "sent": int(row["sent"] or 0),
                "recv": int(row["recv"] or 0),
            }
        sent_data = []
        recv_data = []
        for y, mo in month_keys:
            cell = by_month.get((y, mo), {"sent": 0, "recv": 0})
            sent_data.append(cell["sent"])
            recv_data.append(cell["recv"])
        prov_member = ThirdPartyApiRequestLog.Provider(tp)
        lbl = provider_labels[prov_member]
        col = colors[prov_member]
        sent_datasets = [{"label": lbl, "data": sent_data, "backgroundColor": col}]
        recv_datasets = [{"label": lbl, "data": recv_data, "backgroundColor": col}]

    total_sent = sum(sum(d["data"]) for d in sent_datasets)
    total_recv = sum(sum(d["data"]) for d in recv_datasets)

    return {
        "token_chart_labels": labels,
        "token_sent_datasets": sent_datasets,
        "token_recv_datasets": recv_datasets,
        "tokens_platform_filter": tp,
        "token_total_sent": total_sent,
        "token_total_received": total_recv,
    }


def _month_range_aware(year: int, month: int) -> tuple[datetime, datetime]:
    tz = timezone.get_current_timezone()
    start = timezone.make_aware(datetime(year, month, 1, 0, 0, 0), tz)
    if month == 12:
        end = timezone.make_aware(datetime(year + 1, 1, 1, 0, 0, 0), tz)
    else:
        end = timezone.make_aware(datetime(year, month + 1, 1, 0, 0, 0), tz)
    return start, end


def _platform_chart_label(slug: str) -> str:
    s = (slug or "").strip().lower()
    if s == "openai":
        return "OpenAI"
    if s == "gemini":
        return "Gemini"
    if s == "perplexity":
        return "Perplexity"
    return (slug or "unknown").replace("_", " ").title()


def _visibility_pct_for_profile_month(
    profile: BusinessProfile,
    year: int,
    month: int,
    *,
    response_platform: str | None,
) -> float | None:
    start, end = _month_range_aware(year, month)
    extractions = latest_extraction_per_response_in_window(
        profile,
        start=start,
        end=end,
        response_platform=response_platform,
    )
    if not extractions:
        return None
    site = (getattr(profile, "website_url", None) or "").strip()
    return float(calculate_visibility_score(extractions, tracked_website_url=site))


_PLATFORM_LINE_COLORS: dict[str, str] = {
    "openai": "rgb(16, 163, 127)",
    "gemini": "rgb(66, 133, 244)",
    "perplexity": "rgb(124, 58, 237)",
}


def build_monthly_aeo_visibility_chart_context(
    business_profile_id: int | None,
    *,
    months: int = 12,
) -> dict[str, Any]:
    """
    Monthly AEO brand-visibility % from extractions tied to responses created in each month.

    - **Total**: all platforms (one extraction per response row; prompts with both OpenAI and Gemini
      count twice toward the denominator, matching ``response_platform=None`` semantics).
    - **By platform**: separate series per ``AEOResponseSnapshot.platform`` value seen in the window.

    For **all profiles**, averages visibility across up to 200 profiles that have any AEO response
    in the chart window (keeps the staff home page responsive).
    """
    month_keys = _last_n_calendar_month_starts(months)
    labels = [_month_label(y, mo) for y, mo in month_keys]
    first_y, first_m = month_keys[0]
    chart_start, _ = _month_range_aware(first_y, first_m)

    if business_profile_id is not None:
        profiles = list(
            BusinessProfile.objects.filter(pk=business_profile_id).only("id", "website_url"),
        )
    else:
        ids = list(
            AEOResponseSnapshot.objects.filter(created_at__gte=chart_start)
            .order_by("profile_id")
            .values_list("profile_id", flat=True)
            .distinct()[:200]
        )
        profiles = list(
            BusinessProfile.objects.filter(pk__in=ids).only("id", "website_url").order_by("pk"),
        )

    platform_slugs = sorted(
        {
            str(p).strip().lower()
            for p in AEOResponseSnapshot.objects.filter(created_at__gte=chart_start).values_list(
                "platform",
                flat=True,
            )
            if str(p).strip()
        }
    )
    if not platform_slugs:
        platform_slugs = ["openai", "gemini", "perplexity"]

    total_data: list[float] = []
    for y, mo in month_keys:
        vals: list[float] = []
        for p in profiles:
            v = _visibility_pct_for_profile_month(p, y, mo, response_platform=None)
            if v is not None:
                vals.append(v)
        total_data.append(round(sum(vals) / len(vals), 2) if vals else 0.0)

    by_platform: list[dict[str, Any]] = []
    for slug in platform_slugs:
        series: list[float] = []
        for y, mo in month_keys:
            vals_p: list[float] = []
            for p in profiles:
                v = _visibility_pct_for_profile_month(p, y, mo, response_platform=slug)
                if v is not None:
                    vals_p.append(v)
            series.append(round(sum(vals_p) / len(vals_p), 2) if vals_p else 0.0)
        by_platform.append(
            {
                "platform": slug,
                "label": _platform_chart_label(slug),
                "data": series,
                "borderColor": _PLATFORM_LINE_COLORS.get(
                    slug,
                    "rgb(108, 117, 125)",
                ),
            }
        )

    return {
        "aeo_visibility_labels": labels,
        "aeo_visibility_total": total_data,
        "aeo_visibility_by_platform": by_platform,
    }


def build_aeo_pass_count_analytics_context(
    *,
    execution_run_id: int | None = None,
    profile_id: int | None = None,
) -> dict[str, Any]:
    from accounts.aeo.perplexity_execution_utils import perplexity_execution_enabled

    use_perplexity = perplexity_execution_enabled()

    qs = AEOPromptExecutionAggregate.objects.all()
    if execution_run_id is not None:
        qs = qs.filter(execution_run_id=execution_run_id)
    if profile_id is not None:
        qs = qs.filter(profile_id=profile_id)
    rows = list(qs)
    total_prompts = len(rows)

    def _provider_breakdown(provider: str) -> dict[str, int]:
        if provider == "openai":
            pass_attr = "openai_pass_count"
            stable_attr = "openai_stability_status"
            need_attr = "openai_third_pass_required"
            ran_attr = "openai_third_pass_ran"
        elif provider == "gemini":
            pass_attr = "gemini_pass_count"
            stable_attr = "gemini_stability_status"
            need_attr = "gemini_third_pass_required"
            ran_attr = "gemini_third_pass_ran"
        else:
            pass_attr = "perplexity_pass_count"
            stable_attr = "perplexity_stability_status"
            need_attr = "perplexity_third_pass_required"
            ran_attr = "perplexity_third_pass_ran"
        total = sum(1 for r in rows if int(getattr(r, pass_attr, 0) or 0) > 0)
        stable_2 = sum(
            1
            for r in rows
            if int(getattr(r, pass_attr, 0) or 0) >= 2 and str(getattr(r, stable_attr, "")) == "stable"
        )
        needed_third = sum(1 for r in rows if bool(getattr(r, need_attr, False)))
        third_completed = sum(1 for r in rows if bool(getattr(r, ran_attr, False)))
        return {
            "total": total,
            "stable_at_2": stable_2,
            "needed_third": needed_third,
            "third_completed": third_completed,
        }

    def _prompt_stable_at_2_all_in_scope(r: AEOPromptExecutionAggregate) -> bool:
        """
        Prompt counts as stable-at-2 when every in-scope provider (OpenAI + Gemini always;
        Perplexity too when PERPLEXITY_API_KEY is set) has stability_status == stable.
        """
        o_ok = str(r.openai_stability_status) == "stable"
        g_ok = str(r.gemini_stability_status) == "stable"
        if not use_perplexity:
            return o_ok and g_ok
        return o_ok and g_ok and str(r.perplexity_stability_status) == "stable"

    def _prompt_needs_third(r: AEOPromptExecutionAggregate) -> bool:
        b = bool(r.openai_third_pass_required) or bool(r.gemini_third_pass_required)
        if use_perplexity:
            b = b or bool(r.perplexity_third_pass_required)
        return b

    def _prompt_ran_third(r: AEOPromptExecutionAggregate) -> bool:
        b = bool(r.openai_third_pass_ran) or bool(r.gemini_third_pass_ran)
        if use_perplexity:
            b = b or bool(r.perplexity_third_pass_ran)
        return b

    openai = _provider_breakdown("openai")
    gemini = _provider_breakdown("gemini")
    perplexity = _provider_breakdown("perplexity")
    prompts_requiring_3rd = sum(1 for r in rows if _prompt_needs_third(r))
    prompts_ran_3rd = sum(1 for r in rows if _prompt_ran_third(r))
    prompts_stable_at_2 = sum(1 for r in rows if _prompt_stable_at_2_all_in_scope(r))

    by_run = defaultdict(lambda: {"total": 0, "stable_at_2": 0, "needed_third": 0, "third_completed": 0})
    for r in rows:
        key = str(r.execution_run_id or "none")
        by_run[key]["total"] += 1
        if _prompt_stable_at_2_all_in_scope(r):
            by_run[key]["stable_at_2"] += 1
        if _prompt_needs_third(r):
            by_run[key]["needed_third"] += 1
        if _prompt_ran_third(r):
            by_run[key]["third_completed"] += 1

    run_labels = sorted(by_run.keys(), key=lambda x: (x == "none", x))
    run_grouped = {
        "labels": run_labels,
        "total": [by_run[k]["total"] for k in run_labels],
        "stable_at_2": [by_run[k]["stable_at_2"] for k in run_labels],
        "needed_third": [by_run[k]["needed_third"] for k in run_labels],
        "third_completed": [by_run[k]["third_completed"] for k in run_labels],
    }
    top_competitors: dict[str, int] = defaultdict(int)
    top_citations: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in dict(r.combined_competitor_counts or {}).items():
            try:
                top_competitors[str(k)] += int(v or 0)
            except Exception:
                continue
        for k, v in dict(r.combined_citation_counts or {}).items():
            try:
                top_citations[str(k)] += int(v or 0)
            except Exception:
                continue
    top_competitors_sorted = sorted(top_competitors.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_citations_sorted = sorted(top_citations.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {
        "total_prompts": total_prompts,
        "prompts_stable_at_2": prompts_stable_at_2,
        "prompts_requiring_3rd": prompts_requiring_3rd,
        "prompts_ran_3rd": prompts_ran_3rd,
        "perplexity_analytics_in_scope": use_perplexity,
        "providers": {"openai": openai, "gemini": gemini, "perplexity": perplexity},
        "grouped_by_run": run_grouped,
        "top_competitors": [{"key": k, "count": c} for k, c in top_competitors_sorted],
        "top_citations": [{"key": k, "count": c} for k, c in top_citations_sorted],
    }
