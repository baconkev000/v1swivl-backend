import csv
from datetime import datetime

from django.contrib import admin
from django.http import HttpResponse

from .models import (
    AEOPromptExecutionAggregate,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEORecommendationRun,
    AEOExecutionRun,
    AEOScoreSnapshot,
    BusinessProfile,
    ThirdPartyApiErrorLog,
    ThirdPartyApiRequestLog,
    GoogleSearchConsoleConnection,
    GoogleBusinessProfileConnection,
    GoogleAdsConnection,
    MetaAdsConnection,
    OnboardingOnPageCrawl,
    SEOOverviewSnapshot,
    ReviewsOverviewSnapshot,
    GoogleAdsKeywordIdea,
    AgentConversation,
    AgentMessage,
    ReviewsConversation,
    ReviewsMessage,
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
        "total_pass_count",
        "openai_brand_cited_count",
        "gemini_brand_cited_count",
        "total_brand_cited_count",
        "openai_stability_status",
        "gemini_stability_status",
        "openai_third_pass_required",
        "gemini_third_pass_required",
        "openai_third_pass_ran",
        "gemini_third_pass_ran",
        "stability_status",
        "updated_at",
    )
    list_filter = (
        "stability_status",
        "openai_stability_status",
        "gemini_stability_status",
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


@admin.register(BusinessProfile)
class BusinessProfileAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "business_name",
        "industry",
        "tone_of_voice",
        "created_at",
        "updated_at",
    )
    search_fields = ("user__email", "user__username", "business_name", "industry")
    list_filter = ("industry", "tone_of_voice", "created_at")


@admin.register(GoogleSearchConsoleConnection)
class GoogleSearchConsoleConnectionAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleBusinessProfileConnection)
class GoogleBusinessProfileConnectionAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleAdsConnection)
class GoogleAdsConnectionAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(MetaAdsConnection)
class MetaAdsConnectionAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "expires_at", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


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
    readonly_fields = ("created_at", "updated_at")


@admin.register(ReviewsOverviewSnapshot)
class ReviewsOverviewSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "star_rating", "total_reviews", "response_rate_pct", "last_fetched_at")
    search_fields = ("user__email", "user__username")


@admin.register(SEOOverviewSnapshot)
class SEOOverviewSnapshotAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "period_start", "organic_visitors", "keywords_ranking", "top3_positions", "last_fetched_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleAdsKeywordIdea)
class GoogleAdsKeywordIdeaAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "keyword", "avg_monthly_searches", "competition", "last_fetched_at")
    search_fields = ("user__email", "user__username", "keyword")


@admin.register(AgentConversation)
class AgentConversationAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "agent", "title", "created_at", "updated_at")
    search_fields = ("user__email", "user__username", "title", "agent")


@admin.register(AgentMessage)
class AgentMessageAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "created_at")
    search_fields = ("conversation__user__email", "conversation__user__username", "content")


@admin.register(ReviewsConversation)
class ReviewsConversationAdmin(CsvExportAdminMixin, admin.ModelAdmin):
    list_display = ("id", "user", "title", "created_at", "updated_at")
    search_fields = ("user__email", "user__username", "title")


@admin.register(ReviewsMessage)
class ReviewsMessageAdmin(CsvExportAdminMixin, admin.ModelAdmin):
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

