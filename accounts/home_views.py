"""
Site home page: optional staff-only third-party API usage charts.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from django.contrib import messages
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import gettext as _

from accounts.models import AEOExecutionRun, BusinessProfile
from accounts.tasks import sync_enrich_seo_snapshot_for_profile_task
from accounts.third_party_usage import (
    build_aeo_pass_count_analytics_context,
    build_monthly_aeo_visibility_chart_context,
    build_monthly_api_usage_chart_context,
    build_monthly_token_usage_chart_context,
)

logger = logging.getLogger(__name__)


def site_home(request):
    ctx: dict = {}
    user = request.user
    if user.is_authenticated and user.is_staff:
        raw = request.GET.get("profile", "all")
        profile_pk = None
        if raw not in ("all", "", None):
            try:
                profile_pk = int(raw)
            except (TypeError, ValueError):
                profile_pk = None
        chart = build_monthly_api_usage_chart_context(profile_pk, months=12)
        tokens_platform_raw = (request.GET.get("tokens_platform") or "all").strip().lower()
        token_chart = build_monthly_token_usage_chart_context(
            profile_pk,
            tokens_platform_raw,
            months=12,
        )
        aeo_vis = build_monthly_aeo_visibility_chart_context(profile_pk, months=12)
        ctx["show_api_usage_chart"] = True
        ctx["api_usage_total_requests"] = chart["api_usage_total_requests"]
        ctx["api_usage_total_cost_usd"] = chart["api_usage_total_cost_usd"]
        ctx["business_profiles_for_usage"] = list(
            BusinessProfile.objects.order_by("business_name", "pk").values("id", "business_name")[:500]
        )
        ctx["selected_profile_filter"] = "all" if profile_pk is None else str(profile_pk)
        ctx["selected_tokens_platform_filter"] = token_chart["tokens_platform_filter"]
        ctx["api_usage_chart_payload"] = {
            "labels": chart["api_usage_chart_labels"],
            "countDatasets": chart["api_usage_count_datasets"],
            "costDatasets": chart["api_usage_cost_datasets"],
        }
        ctx["aeo_visibility_chart_payload"] = {
            "labels": aeo_vis["aeo_visibility_labels"],
            "total": aeo_vis["aeo_visibility_total"],
            "byPlatform": aeo_vis["aeo_visibility_by_platform"],
            "totalLegend": str(_("All platforms combined")),
        }
        ctx["token_usage_totals"] = {
            "sent": token_chart["token_total_sent"],
            "received": token_chart["token_total_received"],
        }
        ctx["token_chart_payload"] = {
            "labels": token_chart["token_chart_labels"],
            "sentDatasets": token_chart["token_sent_datasets"],
            "recvDatasets": token_chart["token_recv_datasets"],
        }
    return render(request, "pages/home.html", ctx)


def aeo_pass_count_staff_page(request):
    if not request.user.is_authenticated or not request.user.is_staff:
        return HttpResponseForbidden("Staff only")

    if request.method == "POST":
        from accounts.tasks import (
            try_enqueue_aeo_full_monitored_pipeline,
            try_enqueue_aeo_phase5_recommendations_only,
        )

        action = request.POST.get("action")
        if action == "full_rerun":
            pid_str = (request.POST.get("aeo_full_rerun_profile_id") or "").strip()
            if pid_str in ("", "all"):
                messages.warning(
                    request,
                    "Select a specific business profile (not All) to enqueue a full monitored-prompt pipeline re-run.",
                )
            else:
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    messages.error(request, "Invalid profile id.")
                else:
                    out = try_enqueue_aeo_full_monitored_pipeline(pid, source="staff_aeo_pass_counts")
                    if out.get("queued"):
                        messages.success(
                            request,
                            f"{out.get('message', 'Enqueued.')} Run id {out.get('run_id')}.",
                        )
                    elif out.get("reason") == "duplicate_inflight":
                        messages.warning(request, out.get("message") or "A run is already in progress.")
                    elif out.get("reason") == "no_prompts":
                        messages.warning(request, out.get("message") or "No monitored prompts on profile.")
                    elif not out.get("ok"):
                        messages.error(request, out.get("message") or "Could not enqueue pipeline.")
                    else:
                        messages.info(request, out.get("message") or "Done.")
        elif action == "phase5_recommendations_only":
            pid_str = (request.POST.get("aeo_phase5_rerun_profile_id") or "").strip()
            if pid_str in ("", "all"):
                messages.warning(
                    request,
                    "Select a specific business profile (not All) to re-run Phase 5 recommendations.",
                )
            else:
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    messages.error(request, "Invalid profile id.")
                else:
                    out = try_enqueue_aeo_phase5_recommendations_only(
                        pid, source="staff_aeo_pass_counts"
                    )
                    if out.get("queued"):
                        messages.success(
                            request,
                            f"{out.get('message', 'Enqueued.')} Run id {out.get('run_id')}.",
                        )
                    elif out.get("reason") == "duplicate_inflight":
                        messages.warning(request, out.get("message") or "A run is already in progress.")
                    elif out.get("reason") == "phase5_inflight":
                        messages.warning(
                            request,
                            out.get("message") or "Phase 5 is already running for this execution run.",
                        )
                    elif out.get("reason") in ("recommendations_disabled", "no_eligible_run"):
                        messages.warning(
                            request,
                            out.get("message") or "Could not enqueue Phase 5 recommendations.",
                        )
                    elif out.get("reason") == "no_prompts":
                        messages.warning(request, out.get("message") or "No monitored prompts on profile.")
                    elif not out.get("ok"):
                        messages.error(request, out.get("message") or "Could not enqueue Phase 5.")
                    else:
                        messages.info(request, out.get("message") or "Done.")
        elif action == "seo_snapshot_refresh":
            pid_str = (request.POST.get("seo_snapshot_refresh_profile_id") or "").strip()
            if pid_str in ("", "all"):
                messages.warning(
                    request,
                    "Select a specific business profile (not All) to refresh the SEO snapshot.",
                )
            else:
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    messages.error(request, "Invalid profile id.")
                else:
                    profile = (
                        BusinessProfile.objects.filter(pk=pid)
                        .select_related("user")
                        .first()
                    )
                    if profile is None:
                        messages.error(request, "Business profile not found.")
                    else:
                        site_url = (getattr(profile, "website_url", None) or "").strip()
                        if not site_url:
                            messages.warning(
                                request,
                                "Profile has no website URL; SEO snapshot cannot run.",
                            )
                        else:
                            # Run full ranked + enrich pipeline in Celery so this HTTP request does not block
                            # on DataForSEO (avoids Gunicorn worker abort / SystemExit on long calls or disconnects).
                            try:
                                sync_enrich_seo_snapshot_for_profile_task.delay(int(profile.pk))
                            except Exception:
                                logger.exception(
                                    "[staff aeo pass counts] SEO snapshot refresh queue failed profile_id=%s",
                                    pid,
                                )
                                messages.error(
                                    request,
                                    "Could not queue SEO snapshot refresh (Celery broker unreachable or misconfigured). "
                                    "Check server logs.",
                                )
                            else:
                                messages.success(
                                    request,
                                    "SEO snapshot refresh queued: ranked keywords plus gap/suggested enrichment "
                                    "will run in the background (usually a few minutes). Refresh Keywords or Actions "
                                    "after it completes.",
                                )
        else:
            messages.error(request, "Unknown action.")

        base = reverse("staff-aeo-pass-counts")
        rp = (request.POST.get("redirect_run_id") or "all").strip() or "all"
        pp = (request.POST.get("redirect_profile_id") or "all").strip() or "all"
        return redirect(f"{base}?{urlencode({'run_id': rp, 'profile_id': pp})}")

    run_raw = request.GET.get("run_id")
    profile_raw = request.GET.get("profile_id")
    run_id: int | None = None
    profile_id: int | None = None
    try:
        if run_raw not in (None, "", "all"):
            run_id = int(str(run_raw))
    except (TypeError, ValueError):
        run_id = None
    try:
        if profile_raw not in (None, "", "all"):
            profile_id = int(str(profile_raw))
    except (TypeError, ValueError):
        profile_id = None
    analytics = build_aeo_pass_count_analytics_context(
        execution_run_id=run_id,
        profile_id=profile_id,
    )
    ctx = {
        "analytics": analytics,
        "selected_run_id": "all" if run_id is None else str(run_id),
        "selected_profile_id": "all" if profile_id is None else str(profile_id),
        "business_profiles_for_usage": list(
            BusinessProfile.objects.select_related("user")
            .order_by("business_name", "pk")
            .values("id", "business_name", "user__email", "user__username")[:500]
        ),
        "aeo_runs_for_filter": list(
            AEOExecutionRun.objects.select_related("profile", "profile__user")
            .order_by("-created_at", "-id")
            .values(
                "id",
                "profile_id",
                "profile__business_name",
                "profile__user__email",
                "profile__user__username",
            )[:500]
        ),
    }
    return render(request, "pages/aeo_pass_count_staff.html", ctx)
