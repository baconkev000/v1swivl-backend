from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include
from django.urls import path
from django.views import defaults as default_views
from django.views.generic import TemplateView

from accounts.home_views import aeo_pass_count_staff_page, site_home
from drf_spectacular.views import SpectacularAPIView
from drf_spectacular.views import SpectacularSwaggerView
from rest_framework.authtoken.views import obtain_auth_token

from swivl.users.oauth_callback_views import google_oauth_callback_view, microsoft_oauth_callback_view
from swivl.users.views import google_login_redirect_view, microsoft_login_redirect_view
from accounts import views as accounts_views

urlpatterns = [
    path("", site_home, name="home"),
    path(
        "about/",
        TemplateView.as_view(template_name="pages/about.html"),
        name="about",
    ),
    path("staff/aeo-pass-counts/", aeo_pass_count_staff_page, name="staff-aeo-pass-counts"),
    # Django Admin, use {% url 'admin:index' %}
    path(settings.ADMIN_URL, admin.site.urls),
    # User management
    path("users/", include("swivl.users.urls", namespace="users")),
    path("auth/google/login/", google_login_redirect_view, name="google-login"),
    path("auth/microsoft/login/", microsoft_login_redirect_view, name="microsoft-login"),
    # Shadow allauth OAuth callbacks so successful SPA redirects use ``location.replace``
    # (avoids Back landing on ``/accounts/.../callback/``).
    path("accounts/google/login/callback/", google_oauth_callback_view),
    path("accounts/microsoft/login/callback/", microsoft_oauth_callback_view),
    path("accounts/", include("allauth.urls")),
    # Your stuff: custom urls includes go here
    # ...
    # Media files
    *static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT),
]

# API URLS
urlpatterns += [
    # API base url
    path("api/", include("config.api_router")),
    # DRF auth token
    path("api/auth-token/", obtain_auth_token, name="obtain_auth_token"),
    path("api/auth/status/", accounts_views.auth_status, name="auth-status"),
    path("api/auth/login/", accounts_views.api_auth_login, name="api-auth-login"),
    path("api/auth/register/", accounts_views.api_auth_register, name="api-auth-register"),
    path(
        "api/onboarding/onpage-crawl/",
        accounts_views.onboarding_onpage_crawl_start,
        name="onboarding-onpage-crawl",
    ),
    path(
        "api/onboarding/crawl/latest/",
        accounts_views.onboarding_crawl_latest,
        name="onboarding-crawl-latest",
    ),
    # Business profile for settings page (main profile) + list/create/update profiles
    path("api/business-profile/", accounts_views.business_profile, name="business-profile"),
    path("api/business-profile/team/", accounts_views.business_profile_team, name="business-profile-team"),
    path(
        "api/business-profile/team/<int:user_id>/",
        accounts_views.business_profile_team_member,
        name="business-profile-team-member",
    ),
    path("api/billing/summary/", accounts_views.billing_summary, name="billing-summary"),
    path(
        "api/business-profile/checkout-identity/",
        accounts_views.business_profile_checkout_identity,
        name="business-profile-checkout-identity",
    ),
    path(
        "api/onboarding/local-dev-billing-complete/",
        accounts_views.onboarding_local_dev_billing_complete,
        name="onboarding-local-dev-billing-complete",
    ),
    path("api/business-profiles/", accounts_views.business_profile_list, name="business-profile-list"),
    path("api/business-profiles/<int:pk>/", accounts_views.business_profile_detail, name="business-profile-detail"),
    path("api/seo/refresh-next-steps/", accounts_views.refresh_seo_next_steps, name="seo-refresh-next-steps"),
    # Agent activity feed for dashboard "What your agents did today"
    path("api/activity/", accounts_views.agent_activity_feed, name="agent-activity-feed"),
    # SEO overview metrics for dashboard (Google Search Console powered)
    path("api/seo/overview/", accounts_views.seo_overview, name="seo-overview"),
    # High-Intent Keywords dataset for SEO agent
    path("api/seo/keywords/", accounts_views.seo_keywords, name="seo-keywords"),
    # Debug helper: returns saved top_keywords row for a keyword
    path("api/seo/keyword-debug/", accounts_views.seo_keyword_debug, name="seo-keyword-debug"),
    # Force-refresh SEO snapshot (keywords, rankings, visibility) for main business profile
    path("api/seo/refresh-snapshot/", accounts_views.refresh_seo_snapshot, name="refresh-seo-snapshot"),
    path("api/seo/profile/", accounts_views.seo_profile_data, name="seo-profile-data"),
    path("api/seo/score-history/", accounts_views.seo_score_history_data, name="seo-score-history-data"),
    path("api/aeo/profile/", accounts_views.aeo_profile_data, name="aeo-profile-data"),
    path("api/aeo/prompt-coverage/", accounts_views.aeo_prompt_coverage_data, name="aeo-prompt-coverage-data"),
    path(
        "api/aeo/monitored-prompts/",
        accounts_views.aeo_monitored_prompt_append,
        name="aeo-monitored-prompt-append",
    ),
    path(
        "api/aeo/recommendations/complete/",
        accounts_views.aeo_mark_recommendation_complete,
        name="aeo-mark-recommendation-complete",
    ),
    path(
        "api/actions/generate-page-preview/",
        accounts_views.actions_generate_page_preview,
        name="actions-generate-page-preview",
    ),
    path(
        "api/aeo/platform-visibility/",
        accounts_views.aeo_platform_visibility_data,
        name="aeo-platform-visibility-data",
    ),
    path("api/aeo/share-of-voice/", accounts_views.aeo_share_of_voice_data, name="aeo-share-of-voice-data"),
    path("api/aeo/competitors/", accounts_views.aeo_competitors_data, name="aeo-competitors-data"),
    path(
        "api/aeo/onboarding-competitors/",
        accounts_views.aeo_onboarding_competitors_data,
        name="aeo-onboarding-competitors-data",
    ),
    path("api/aeo/pipeline-status/", accounts_views.aeo_pipeline_status_data, name="aeo-pipeline-status-data"),
    path("api/staff/aeo-pass-counts/", accounts_views.aeo_pass_count_analytics_data, name="aeo-pass-count-analytics"),
    path("api/aeo/refresh-snapshot/", accounts_views.refresh_aeo_snapshot, name="refresh-aeo-snapshot"),
    path("api/aeo/refresh-gemini/", accounts_views.refresh_aeo_gemini, name="refresh-aeo-gemini"),
    path(
        "api/aeo/refresh-perplexity/",
        accounts_views.refresh_aeo_perplexity,
        name="refresh-aeo-perplexity",
    ),
    path(
        "api/aeo/retry-prompt-expansion/",
        accounts_views.aeo_retry_prompt_expansion,
        name="aeo-retry-prompt-expansion",
    ),
    path(
        "api/aeo/refresh-execution/",
        accounts_views.aeo_refresh_execution,
        name="aeo-refresh-execution",
    ),
    path(
        "api/aeo/onboarding-prompt-plan/",
        accounts_views.aeo_onboarding_prompt_plan,
        name="aeo-onboarding-prompt-plan",
    ),
    path("api/stripe/webhook/", accounts_views.stripe_webhook, name="stripe-webhook"),
    # SEO agent chat
    path("api/seo/chat/", accounts_views.seo_chat, name="seo-chat"),
    # API logout to clear Django/Google SSO session
    path("api/logout/", accounts_views.api_logout, name="api-logout"),
    path("api/schema/", SpectacularAPIView.as_view(), name="api-schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="api-schema"),
        name="api-docs",
    ),
]

if settings.DEBUG:
    # This allows the error pages to be debugged during development, just visit
    # these url in browser to see how these error pages look like.
    urlpatterns += [
        path(
            "400/",
            default_views.bad_request,
            kwargs={"exception": Exception("Bad Request!")},
        ),
        path(
            "403/",
            default_views.permission_denied,
            kwargs={"exception": Exception("Permission Denied")},
        ),
        path(
            "404/",
            default_views.page_not_found,
            kwargs={"exception": Exception("Page not Found")},
        ),
        path("500/", default_views.server_error),
    ]
    if "debug_toolbar" in settings.INSTALLED_APPS:
        import debug_toolbar

        urlpatterns = [
            path("__debug__/", include(debug_toolbar.urls)),
            *urlpatterns,
        ]
