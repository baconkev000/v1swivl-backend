import csv
import json
from datetime import datetime

from django import forms
from django.contrib import admin, messages
from django.core.exceptions import ValidationError
from django.forms.models import BaseInlineFormSet
from django.http import HttpResponse
from django.urls import reverse
from django.utils.html import escape, format_html

from .models import (
    AEOPromptExecutionAggregate,
    AEOCompetitorSnapshot,
    AEODashboardBundleCache,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEORecommendationRun,
    AEOExecutionRun,
    AEOScoreSnapshot,
    BusinessProfile,
    BusinessProfileMembership,
    TrackedCompetitor,
    ThirdPartyApiErrorLog,
    ThirdPartyApiRequestLog,
    OnboardingOnPageCrawl,
    SEOOverviewSnapshot,
    AgentConversation,
    AgentMessage,
)


class CsvExportAdminMixin:
    """
    Adds a generic "export selected rows to CSV" action.
    """

    actions = ("export_as_csv",)

    @admin.action(description="Export selected rows to CSV")
    def export_as_csv(self, request, queryset):
        model = self.model
        field_names = [field.name for field in model._meta.fields]

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{model._meta.model_name}_{ts}.csv"

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'

        writer = csv.writer(response)
        writer.writerow(field_names)

        for obj in queryset:
            writer.writerow([getattr(obj, field) for field in field_names])

        return response


@admin.register(AEORecommendationRun)
class AEORecommendationRunAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "profile",
        "score_snapshot",
        "visibility_score_at_run",
        "citation_share_at_run",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("profile__business_name",)
    raw_id_fields = ("profile", "score_snapshot")


@admin.register(AEOCompetitorSnapshot)
class AEOCompetitorSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "profile",
        "platform_scope",
        "total_slots",
        "window_start",
        "window_end",
        "updated_at",
        "created_at",
    )
    list_filter = ("platform_scope", "created_at", "updated_at")
    search_fields = ("profile__business_name", "profile__user__email")
    autocomplete_fields = ("profile",)
    date_hierarchy = "updated_at"
    ordering = ("-updated_at",)
    exclude = ("rows_json",)
    readonly_fields = ("created_at", "updated_at", "rows_json_preview")

    @admin.display(description="rows_json (read-only preview)")
    def rows_json_preview(self, obj: AEOCompetitorSnapshot) -> str:
        if obj is None or not getattr(obj, "pk", None):
            return "—"
        data = obj.rows_json
        if data in (None, [], {}):
            return "—"
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(data)
        max_len = 12000
        if len(text) > max_len:
            text = text[:max_len] + "\n… (truncated)"
        return format_html(
            '<pre style="max-height:28rem;overflow:auto;font-size:11px;margin:0;">{}</pre>',
            text,
        )


@admin.register(AEODashboardBundleCache)
class AEODashboardBundleCacheAdmin(admin.ModelAdmin):
    list_display = ("id", "profile", "updated_at")
    search_fields = ("profile__business_name", "profile__user__email")
    autocomplete_fields = ("profile",)
    readonly_fields = ("updated_at", "payload_preview")
    exclude = ("payload_json",)

    @admin.display(description="payload_json (preview)")
    def payload_preview(self, obj: AEODashboardBundleCache) -> str:
        if obj is None or not getattr(obj, "pk", None):
            return "—"
        data = obj.payload_json
        if data in (None, [], {}):
            return "—"
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(data)
        max_len = 8000
        if len(text) > max_len:
            text = text[:max_len] + "\n… (truncated)"
        return format_html(
            '<pre style="max-height:24rem;overflow:auto;font-size:11px;margin:0;">{}</pre>',
            text,
        )


@admin.register(AEOScoreSnapshot)
class AEOScoreSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "profile",
        "visibility_score",
        "weighted_position_score",
        "citation_share",
        "total_prompts",
        "total_mentions",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("profile__business_name",)
    raw_id_fields = ("profile",)


@admin.register(AEOExtractionSnapshot)
class AEOExtractionSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "response_snapshot",
        "brand_mentioned",
        "brand_mentioned_url_status",
        "cited_domain_or_url",
        "mention_position",
        "mention_count",
        "sentiment",
        "extraction_parse_failed",
        "extraction_model",
        "created_at",
    )
    list_filter = (
        "brand_mentioned",
        "brand_mentioned_url_status",
        "mention_position",
        "sentiment",
        "extraction_parse_failed",
    )
    search_fields = ("response_snapshot__prompt_hash", "response_snapshot__prompt_text")
    raw_id_fields = ("response_snapshot",)


@admin.register(AEOResponseSnapshot)
class AEOResponseSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("profile_business_name", "platform")
    list_filter = ("platform", "prompt_type", "is_dynamic")
    search_fields = ("prompt_hash", "prompt_text", "profile__business_name", "execution_pair_id")
    raw_id_fields = ("profile", "execution_run")

    @admin.display(description="Profile", ordering="profile__business_name")
    def profile_business_name(self, obj: AEOResponseSnapshot) -> str:
        p = obj.profile
        if p is None:
            return "—"
        name = (p.business_name or "").strip()
        return name or f"Profile #{p.pk}"


@admin.register(AEOExecutionRun)
class AEOExecutionRunAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    actions = (
        "export_as_csv",
        "queue_retry_missing_phase3_extractions",
        "queue_retry_missing_phase3_extractions_perplexity",
    )
    list_display = (
        "id",
        "profile",
        "status",
        "fetch_mode",
        "prompt_count_requested",
        "prompt_count_executed",
        "prompt_count_failed",
        "background_status",
        "phase1_provider_calls",
        "phase1_completed_at",
        "extraction_count",
        "extraction_status",
        "scoring_status",
        "created_at",
        "finished_at",
    )
    list_filter = ("status", "fetch_mode", "background_status", "extraction_status", "scoring_status", "created_at")
    search_fields = ("profile__business_name", "profile__user__email")
    raw_id_fields = ("profile",)

    @admin.action(description="AEO: queue Phase-3 retry for snapshots missing extractions (slow / 1 worker)")
    def queue_retry_missing_phase3_extractions(self, request, queryset):
        from .tasks import retry_aeo_missing_extractions_for_run_task

        n = 0
        for run in queryset:
            retry_aeo_missing_extractions_for_run_task.delay(run.id, None, 1)
            n += 1
        self.message_user(
            request,
            f"Queued Phase-3 extraction retry for {n} run(s) (all platforms missing extractions, max 1 worker).",
            messages.SUCCESS,
        )

    @admin.action(description="AEO: queue Phase-3 retry — Perplexity only (rate-limit friendly)")
    def queue_retry_missing_phase3_extractions_perplexity(self, request, queryset):
        from .tasks import retry_aeo_missing_extractions_for_run_task

        n = 0
        for run in queryset:
            retry_aeo_missing_extractions_for_run_task.delay(run.id, "perplexity", 1)
            n += 1
        self.message_user(
            request,
            f"Queued Perplexity-only Phase-3 retry for {n} run(s).",
            messages.SUCCESS,
        )


@admin.register(AEOPromptExecutionAggregate)
class AEOPromptExecutionAggregateAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "profile",
        "execution_run",
        "short_prompt_hash",
        "prompt_category",
        "openai_pass_count",
        "gemini_pass_count",
        "perplexity_pass_count",
        "total_pass_count",
        "openai_brand_cited_count",
        "gemini_brand_cited_count",
        "perplexity_brand_cited_count",
        "total_brand_cited_count",
        "openai_stability_status",
        "gemini_stability_status",
        "perplexity_stability_status",
        "openai_third_pass_required",
        "gemini_third_pass_required",
        "perplexity_third_pass_required",
        "openai_third_pass_ran",
        "gemini_third_pass_ran",
        "perplexity_third_pass_ran",
        "combined_total_unique_competitors",
        "combined_total_unique_citations",
        "combined_total_passes_observed",
        "stability_status",
        "updated_at",
    )
    list_filter = (
        "stability_status",
        "openai_stability_status",
        "gemini_stability_status",
        "perplexity_stability_status",
        "prompt_category",
        "execution_run",
        "created_at",
        "updated_at",
    )
    search_fields = (
        "profile__business_name",
        "prompt_text",
        "prompt_hash",
    )
    raw_id_fields = ("profile", "execution_run")
    readonly_fields = (
        "created_at",
        "updated_at",
    )

    @admin.display(description="Prompt hash")
    def short_prompt_hash(self, obj: AEOPromptExecutionAggregate) -> str:
        return (obj.prompt_hash or "")[:12]


class BusinessProfileMembershipInlineForm(forms.ModelForm):
    """Same rules as ``business_profile_team`` POST where applicable."""

    class Meta:
        model = BusinessProfileMembership
        fields = ("user", "role", "is_owner", "hidden_from_team_ui")

    def clean(self):
        cleaned = super().clean()
        if self.errors:
            return cleaned
        profile = cleaned.get("business_profile") or getattr(self.instance, "business_profile", None)
        user = cleaned.get("user")
        role = cleaned.get("role")
        is_owner = cleaned.get("is_owner")
        if profile is None or user is None:
            return cleaned

        if user.pk == profile.user_id:
            if not is_owner:
                raise ValidationError(
                    {
                        "is_owner": (
                            "The account holder (BusinessProfile.user) must have is_owner=True; "
                            "do not add them as a non-owner team row."
                        ),
                    },
                )
            if role != BusinessProfileMembership.ROLE_ADMIN:
                raise ValidationError(
                    {"role": "The owner row must use role “admin”."},
                )
            if cleaned.get("hidden_from_team_ui"):
                raise ValidationError(
                    {"hidden_from_team_ui": "The owner row cannot be hidden from team UI."},
                )
        else:
            if is_owner:
                raise ValidationError(
                    {
                        "is_owner": (
                            "Only BusinessProfile.user may be marked as owner; "
                            "team admins and members must have is_owner=False."
                        ),
                    },
                )

        dup = BusinessProfileMembership.objects.filter(business_profile=profile, user=user)
        if self.instance.pk:
            dup = dup.exclude(pk=self.instance.pk)
        if dup.exists():
            raise ValidationError({"user": "This user already has a membership on this profile."})

        return cleaned


class BusinessProfileMembershipInlineFormSet(BaseInlineFormSet):
    def clean(self):
        super().clean()
        owner_rows = 0
        for form in self.forms:
            if not hasattr(form, "cleaned_data"):
                continue
            cd = form.cleaned_data
            if not cd:
                continue
            if cd.get("DELETE") and getattr(form.instance, "pk", None) and form.instance.is_owner:
                raise ValidationError(
                    "Cannot delete the primary owner (is_owner=True) membership row.",
                )
            if cd.get("DELETE"):
                continue
            if cd.get("is_owner"):
                owner_rows += 1
        if owner_rows > 1:
            raise ValidationError(
                "Only one team membership may have is_owner=True (primary billing owner) per profile.",
            )


class BusinessProfileMembershipInline(admin.TabularInline):
    model = BusinessProfileMembership
    fk_name = "business_profile"
    form = BusinessProfileMembershipInlineForm
    formset = BusinessProfileMembershipInlineFormSet
    extra = 0
    raw_id_fields = ("user",)
    fields = ("user", "role", "is_owner", "hidden_from_team_ui", "created_at", "updated_at")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-is_owner", "role", "id")
    show_change_link = True
    verbose_name_plural = "Team memberships"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")


class ThirdPartyApiErrorLogInline(admin.TabularInline):
    model = ThirdPartyApiErrorLog
    fk_name = "business_profile"
    extra = 0
    can_delete = False
    show_change_link = True
    fields = ("created_at", "provider", "operation", "error_kind", "http_status", "message")
    readonly_fields = fields
    ordering = ("-created_at",)
    verbose_name_plural = "Third-party API errors (this profile; newest first via model ordering)"

    # Cap rows so the change-form POST stays under Django's DATA_UPLOAD_MAX_NUMBER_FIELDS.
    _max_rows = 120

    def get_queryset(self, request):
        base = super().get_queryset(request).order_by("-created_at")
        pks = list(base.values_list("pk", flat=True)[: self._max_rows])
        if not pks:
            return base.none()
        return base.model.objects.filter(pk__in=pks).order_by("-created_at")


@admin.register(BusinessProfile)
class BusinessProfileAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    actions = (
        "export_as_csv",
        "queue_aeo_latest_run_phase3_retry_missing",
        "queue_aeo_latest_run_phase3_retry_missing_perplexity",
    )
    list_display = (
        "id",
        "user",
        "business_name",
        "industry",
        "plan",
        "aeo_prompt_expansion_status",
        "aeo_expansion_progress_display",
        "aeo_prompt_expansion_updated_at",
        "created_at",
        "updated_at",
    )
    search_fields = ("user__email", "user__username", "business_name", "industry")
    list_filter = ("industry", "plan", "aeo_prompt_expansion_status", "created_at")
    filter_horizontal = ("tracked_competitors",)
    inlines = (
        BusinessProfileMembershipInline,
        ThirdPartyApiErrorLogInline,
    )
    readonly_fields = (
        "created_at",
        "updated_at",
        "aeo_latest_run_missing_extractions_display",
    )
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "user",
                    "is_main",
                    "full_name",
                    "business_name",
                    "business_address",
                    "industry",
                    "phone",
                    "description",
                    "website_url",
                ),
            },
        ),
        (
            "Billing",
            {
                "fields": (
                    "plan",
                    "stripe_customer_id",
                    "stripe_subscription_id",
                    "stripe_price_id",
                    "stripe_subscription_status",
                    "stripe_current_period_end",
                    "stripe_cancel_at_period_end",
                ),
            },
        ),
        (
            "SEO & competitors",
            {
                "fields": (
                    "seo_competitor_domains_override",
                    "tracked_competitors",
                    "seo_location_mode",
                    "seo_location_depth",
                    "seo_location_code",
                    "seo_location_label",
                ),
            },
        ),
        (
            "AEO monitored prompts",
            {
                "fields": (
                    "selected_aeo_prompts",
                    "aeo_latest_run_missing_extractions_display",
                ),
            },
        ),
        (
            "AEO prompt expansion (plan-based monitored prompt growth)",
            {
                "description": (
                    "Updated by the post-payment Celery task when Pro/Advanced adds prompts. "
                    "last_error is set on partial runs or failures."
                ),
                "fields": (
                    "aeo_prompt_expansion_status",
                    "aeo_prompt_expansion_target",
                    "aeo_prompt_expansion_progress",
                    "aeo_prompt_expansion_last_error",
                    "aeo_prompt_expansion_updated_at",
                ),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at")},
        ),
    )

    @admin.display(description="AEO expansion (count)")
    def aeo_expansion_progress_display(self, obj: BusinessProfile) -> str:
        if obj is None:
            return "—"
        t = obj.aeo_prompt_expansion_target
        p = obj.aeo_prompt_expansion_progress
        if t is not None:
            return f"{p} / {t}"
        return str(p)

    @admin.display(description="Latest AEO run — response snapshots missing Phase-3 extraction")
    def aeo_latest_run_missing_extractions_display(self, obj: BusinessProfile) -> str:
        if not obj.pk:
            return "—"
        from .aeo.extraction_retry import list_aeo_response_snapshot_ids_missing_extractions

        run = AEOExecutionRun.objects.filter(profile=obj).order_by("-created_at", "-id").first()
        if run is None:
            return "No AEO execution runs for this profile."
        missing = list_aeo_response_snapshot_ids_missing_extractions(run.id)
        missing_p = list_aeo_response_snapshot_ids_missing_extractions(run.id, platform="perplexity")
        url = reverse("admin:accounts_aeoexecutionrun_change", args=[run.id])
        if not missing:
            return format_html(
                'None missing on latest run <a href="{}">#{}</a>.',
                url,
                run.id,
            )
        return format_html(
            '<strong>{}</strong> snapshot(s) missing any extraction on run <a href="{}">#{}</a> '
            "({} Perplexity). On the changelist, use action "
            "<em>AEO: queue Phase-3 retry on latest run (missing extractions)</em>.",
            len(missing),
            url,
            run.id,
            len(missing_p),
        )

    @admin.action(description="AEO: queue Phase-3 retry on latest run (missing extractions, slow)")
    def queue_aeo_latest_run_phase3_retry_missing(self, request, queryset):
        from .tasks import retry_aeo_missing_extractions_for_run_task

        queued = 0
        skipped = 0
        for profile in queryset:
            run = AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
            if run is None:
                skipped += 1
                continue
            retry_aeo_missing_extractions_for_run_task.delay(run.id, None, 1)
            queued += 1
        self.message_user(
            request,
            f"Queued Phase-3 extraction retry for latest run on {queued} profile(s). "
            f"Skipped profiles with no runs: {skipped}.",
            messages.SUCCESS,
        )

    @admin.action(description="AEO: queue Phase-3 retry on latest run — Perplexity missing only")
    def queue_aeo_latest_run_phase3_retry_missing_perplexity(self, request, queryset):
        from .tasks import retry_aeo_missing_extractions_for_run_task

        queued = 0
        skipped = 0
        for profile in queryset:
            run = AEOExecutionRun.objects.filter(profile=profile).order_by("-created_at", "-id").first()
            if run is None:
                skipped += 1
                continue
            retry_aeo_missing_extractions_for_run_task.delay(run.id, "perplexity", 1)
            queued += 1
        self.message_user(
            request,
            f"Queued Perplexity-only Phase-3 retry for latest run on {queued} profile(s). "
            f"Skipped: {skipped}.",
            messages.SUCCESS,
        )


@admin.register(TrackedCompetitor)
class TrackedCompetitorAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "name", "domain", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("name", "domain")
    date_hierarchy = "created_at"
    ordering = ("domain",)


@admin.register(OnboardingOnPageCrawl)
class OnboardingOnPageCrawlAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "business_profile",
        "domain",
        "status",
        "max_pages",
        "task_id",
        "prompt_plan_status",
        "prompt_plan_prompt_count",
        "exit_reason",
        "created_at",
    )
    list_filter = ("status", "prompt_plan_status", "created_at")
    search_fields = (
        "domain",
        "user__email",
        "user__username",
        "business_profile__business_name",
        "task_id",
        "exit_reason",
        "prompt_plan_status",
        "prompt_plan_task_id",
    )
    raw_id_fields = ("user", "business_profile")
    readonly_fields = ("created_at", "updated_at", "review_topics", "review_topics_error")


@admin.register(SEOOverviewSnapshot)
class SEOOverviewSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "business_profile",
        "period_start",
        "organic_visitors",
        "keywords_ranking",
        "top3_positions",
        "has_seo_structured_issues",
        "seo_structured_issues_refreshed_at",
        "last_fetched_at",
    )
    search_fields = (
        "user__email",
        "user__username",
        "business_profile__business_name",
    )
    raw_id_fields = ("user", "business_profile")
    readonly_fields = (
        "last_fetched_at",
        "seo_structured_issues_refreshed_at",
        "seo_structured_issues_formatted",
    )
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "user",
                    "business_profile",
                    "period_start",
                    "last_fetched_at",
                    "refreshed_at",
                    "cached_domain",
                    "cached_location_mode",
                    "cached_location_code",
                    "cached_location_label",
                    "organic_visitors",
                    "prev_organic_visitors",
                    "keywords_ranking",
                    "top3_positions",
                    "total_search_volume",
                    "estimated_search_appearances_monthly",
                    "missed_searches_monthly",
                    "search_visibility_percent",
                    "search_performance_score",
                    "local_verification_applied",
                    "local_verified_keyword_count",
                    "keywords_enriched_at",
                    "seo_next_steps_refreshed_at",
                    "keyword_action_suggestions_refreshed_at",
                    "seo_structured_issues_refreshed_at",
                )
            },
        ),
        (
            "Structured SEO issues (read-only)",
            {
                "classes": ("collapse",),
                "fields": ("seo_structured_issues_formatted",),
            },
        ),
        (
            "JSON payloads",
            {
                "classes": ("collapse",),
                "fields": ("top_keywords", "seo_next_steps", "keyword_action_suggestions"),
            },
        ),
    )

    @admin.display(description="Has structured issues", boolean=True)
    def has_seo_structured_issues(self, obj):
        return bool(getattr(obj, "seo_structured_issues", None))

    @admin.display(description="Structured issues (JSON)")
    def seo_structured_issues_formatted(self, obj):
        raw = getattr(obj, "seo_structured_issues", None) or []
        if not raw:
            return "—"
        try:
            text = json.dumps(raw, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(raw)
        if len(text) > 50000:
            text = text[:50000] + "\n… (truncated)"
        return format_html(
            '<pre style="max-height:480px;overflow:auto;white-space:pre-wrap;">{}</pre>',
            escape(text),
        )


@admin.register(AgentConversation)
class AgentConversationAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "agent", "title", "created_at", "updated_at")
    search_fields = ("user__email", "user__username", "title", "agent")


@admin.register(AgentMessage)
class AgentMessageAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "created_at")
    search_fields = ("conversation__user__email", "conversation__user__username", "content")


@admin.register(ThirdPartyApiErrorLog)
class ThirdPartyApiErrorLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "provider",
        "error_kind",
        "http_status",
        "business_profile",
        "operation",
        "message",
        "created_at",
    )
    list_filter = ("provider", "error_kind", "created_at")
    search_fields = ("operation", "message", "business_profile__business_name")
    raw_id_fields = ("business_profile",)
    readonly_fields = (
        "created_at",
        "provider",
        "business_profile",
        "operation",
        "http_status",
        "error_kind",
        "message",
        "detail",
    )

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(ThirdPartyApiRequestLog)
class ThirdPartyApiRequestLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "provider",
        "business_profile",
        "operation",
        "tokens_sent",
        "tokens_received",
        "cost_usd",
        "created_at",
    )
    list_filter = ("provider", "created_at")
    search_fields = ("operation", "business_profile__business_name")
    raw_id_fields = ("business_profile",)
    readonly_fields = (
        "provider",
        "business_profile",
        "operation",
        "tokens_sent",
        "tokens_received",
        "cost_usd",
        "created_at",
    )

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

