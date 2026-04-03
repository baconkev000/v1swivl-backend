"""
Site home page: optional staff-only third-party API usage charts.
"""

from __future__ import annotations

from django.shortcuts import render
from django.utils.translation import gettext as _

from accounts.models import BusinessProfile
from accounts.third_party_usage import (
    build_monthly_aeo_visibility_chart_context,
    build_monthly_api_usage_chart_context,
    build_monthly_token_usage_chart_context,
)


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
