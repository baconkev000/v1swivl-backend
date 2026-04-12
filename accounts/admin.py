import csv
import json
from datetime import datetime

from django.contrib import admin
from django.http import HttpResponse
from django.utils.html import format_html

from .models import (
    AEOPromptExecutionAggregate,
    AEOCompetitorSnapshot,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEORecommendationRun,
    AEOExecutionRun,
    AEOScoreSnapshot,
    BusinessProfile,
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


class ThirdPartyApiRequestLogInline(admin.TabularInline):
    model = ThirdPartyApiRequestLog
    fk_name = "business_profile"
    extra = 0
    can_delete = False
    show_change_link = True
    fields = ("created_at", "provider", "operation", "tokens_sent", "tokens_received", "cost_usd")
    readonly_fields = fields
    ordering = ("-created_at",)
    verbose_name_plural = "Third-party API requests (this profile; newest first via model ordering)"

    def get_queryset(self, request):
        # Must not slice here: the inline formset applies FK filters after get_queryset().
        return super().get_queryset(request).order_by("-created_at")


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

    def get_queryset(self, request):
        return super().get_queryset(request).order_by("-created_at")


@admin.register(BusinessProfile)
class BusinessProfileAdmin(CsvExportAdminMixin, admin.ModelAdmin):
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
    inlines = (ThirdPartyApiRequestLogInline, ThirdPartyApiErrorLogInline)
    readonly_fields = ("created_at", "updated_at")
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
                "fields": ("selected_aeo_prompts",),
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
    list_display = ("id", "user", "period_start", "organic_visitors", "keywords_ranking", "top3_positions", "last_fetched_at")
    search_fields = ("user__email", "user__username")


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

