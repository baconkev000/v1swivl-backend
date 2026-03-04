from django.contrib import admin

from .models import (
    BusinessProfile,
    GoogleSearchConsoleConnection,
    GoogleAdsConnection,
    SEOOverviewSnapshot,
    GoogleAdsKeywordIdea,
    AgentConversation,
    AgentMessage,
)


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


@admin.register(GoogleAdsConnection)
class GoogleAdsConnectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "updated_at")
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

