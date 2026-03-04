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
    # Business profile for settings page
    path("api/business-profile/", accounts_views.business_profile, name="business-profile"),
    # Google Search Console integration status
    path("api/integrations/google-search-console/status/", accounts_views.gsc_status, name="gsc-status"),
    # Google Ads integration status
    path("api/integrations/google-ads/status/", accounts_views.ads_status, name="gads-status"),
    # SEO overview metrics for dashboard (Google Search Console powered)
    path("api/seo/overview/", accounts_views.seo_overview, name="seo-overview"),
    # High-Intent Keywords dataset for SEO agent
    path("api/seo/keywords/", accounts_views.seo_keywords, name="seo-keywords"),
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
