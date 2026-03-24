from django.contrib import admin

from .models import (
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    AEORecommendationRun,
    AEOScoreSnapshot,
    BusinessProfile,
    GoogleSearchConsoleConnection,
    GoogleBusinessProfileConnection,
    GoogleAdsConnection,
    MetaAdsConnection,
    SEOOverviewSnapshot,
    ReviewsOverviewSnapshot,
    GoogleAdsKeywordIdea,
    AgentConversation,
    AgentMessage,
    ReviewsConversation,
    ReviewsMessage,
)


@admin.register(AEORecommendationRun)
class AEORecommendationRunAdmin(admin.ModelAdmin):
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
class AEOScoreSnapshotAdmin(admin.ModelAdmin):
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
class AEOExtractionSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "response_snapshot",
        "brand_mentioned",
        "mention_position",
        "mention_count",
        "sentiment",
        "extraction_parse_failed",
        "extraction_model",
        "created_at",
    )
    list_filter = ("brand_mentioned", "mention_position", "sentiment", "extraction_parse_failed")
    search_fields = ("response_snapshot__prompt_hash", "response_snapshot__prompt_text")
    raw_id_fields = ("response_snapshot",)


@admin.register(AEOResponseSnapshot)
class AEOResponseSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "profile",
        "prompt_hash",
        "prompt_type",
        "model_name",
        "platform",
        "created_at",
    )
    list_filter = ("platform", "prompt_type", "is_dynamic")
    search_fields = ("prompt_hash", "prompt_text", "profile__business_name")
    raw_id_fields = ("profile",)


@admin.register(BusinessProfile)
class BusinessProfileAdmin(admin.ModelAdmin):
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
class GoogleSearchConsoleConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleBusinessProfileConnection)
class GoogleBusinessProfileConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleAdsConnection)
class GoogleAdsConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(MetaAdsConnection)
class MetaAdsConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "expires_at", "created_at", "updated_at")
    search_fields = ("user__email", "user__username")


@admin.register(ReviewsOverviewSnapshot)
class ReviewsOverviewSnapshotAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "star_rating", "total_reviews", "response_rate_pct", "last_fetched_at")
    search_fields = ("user__email", "user__username")


@admin.register(SEOOverviewSnapshot)
class SEOOverviewSnapshotAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "period_start", "organic_visitors", "keywords_ranking", "top3_positions", "last_fetched_at")
    search_fields = ("user__email", "user__username")


@admin.register(GoogleAdsKeywordIdea)
class GoogleAdsKeywordIdeaAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "keyword", "avg_monthly_searches", "competition", "last_fetched_at")
    search_fields = ("user__email", "user__username", "keyword")


@admin.register(AgentConversation)
class AgentConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "agent", "title", "created_at", "updated_at")
    search_fields = ("user__email", "user__username", "title", "agent")


@admin.register(AgentMessage)
class AgentMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "created_at")
    search_fields = ("conversation__user__email", "conversation__user__username", "content")


@admin.register(ReviewsConversation)
class ReviewsConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "created_at", "updated_at")
    search_fields = ("user__email", "user__username", "title")


@admin.register(ReviewsMessage)
class ReviewsMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "role", "created_at")
    search_fields = ("conversation__user__email", "conversation__user__username", "content")

