from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include
from django.urls import path
from django.views import defaults as default_views
from django.views.generic import TemplateView
from drf_spectacular.views import SpectacularAPIView
from drf_spectacular.views import SpectacularSwaggerView
from rest_framework.authtoken.views import obtain_auth_token

from swivl.users.views import google_login_redirect_view
from accounts import views as accounts_views

urlpatterns = [
    path("", TemplateView.as_view(template_name="pages/home.html"), name="home"),
    path(
        "about/",
        TemplateView.as_view(template_name="pages/about.html"),
        name="about",
    ),
    # Django Admin, use {% url 'admin:index' %}
    path(settings.ADMIN_URL, admin.site.urls),
    # User management
    path("users/", include("swivl.users.urls", namespace="users")),
    path("auth/google/login/", google_login_redirect_view, name="google-login"),
    path(
        "integrations/google-search-console/start/",
        accounts_views.gsc_connect_start,
        name="gsc-connect-start",
    ),
    path(
        "integrations/google-search-console/callback/",
        accounts_views.gsc_connect_callback,
        name="gsc-connect-callback",
    ),
    path(
        "integrations/google-ads/start/",
        accounts_views.ads_connect_start,
        name="gads-connect-start",
    ),
    path(
        "integrations/google-ads/callback/",
        accounts_views.ads_connect_callback,
        name="gads-connect-callback",
    ),
    path(
        "integrations/google-business-profile/start/",
        accounts_views.gbp_connect_start,
        name="gbp-connect-start",
    ),
    path(
        "integrations/google-business-profile/callback/",
        accounts_views.gbp_connect_callback,
        name="gbp-connect-callback",
    ),
    path(
        "integrations/meta-ads/start/",
        accounts_views.meta_ads_connect_start,
        name="meta-ads-connect-start",
    ),
    path(
        "integrations/meta-ads/callback/",
        accounts_views.meta_ads_connect_callback,
        name="meta-ads-connect-callback",
    ),
    path(
        "integrations/tiktok-ads/start/",
        accounts_views.tiktok_ads_connect_start,
        name="tiktok-ads-connect-start",
    ),
    path(
        "integrations/tiktok-ads/callback/",
        accounts_views.tiktok_ads_connect_callback,
        name="tiktok-ads-connect-callback",
    ),
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
    # Business profile for settings page (main profile) + list/create/update profiles
    path("api/business-profile/", accounts_views.business_profile, name="business-profile"),
    path("api/business-profiles/", accounts_views.business_profile_list, name="business-profile-list"),
    path("api/business-profiles/<int:pk>/", accounts_views.business_profile_detail, name="business-profile-detail"),
    path("api/seo/refresh-next-steps/", accounts_views.refresh_seo_next_steps, name="seo-refresh-next-steps"),
    # Google Search Console integration status
    path("api/integrations/google-search-console/status/", accounts_views.gsc_status, name="gsc-status"),
    # Google Ads integration status
    path("api/integrations/google-ads/status/", accounts_views.ads_status, name="gads-status"),
    # Meta Ads integration status
    path("api/integrations/meta-ads/status/", accounts_views.meta_ads_status, name="meta-ads-status"),
    # TikTok Ads integration status
    path("api/integrations/tiktok-ads/status/", accounts_views.tiktok_ads_status, name="tiktok-ads-status"),
    # Google Ads performance metrics (conversions, ROAS, cost per customer)
    # Google Business Profile integration status (Reviews Agent)
    path("api/integrations/google-business-profile/status/", accounts_views.gbp_status, name="gbp-status"),
    # Reviews overview (star rating, total reviews, response rate, etc. — GBP or cached)
    path("api/reviews/overview/", accounts_views.reviews_overview, name="reviews-overview"),
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
    # SEO agent chat
    path("api/seo/chat/", accounts_views.seo_chat, name="seo-chat"),
    # Reviews agent chat (separate tables, different system role)
    path("api/reviews/chat/", accounts_views.reviews_chat, name="reviews-chat"),
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
