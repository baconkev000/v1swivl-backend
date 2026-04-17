# ruff: noqa: ERA001, E501
"""Base settings to build other settings files upon."""

from pathlib import Path
import os
import environ
from celery.schedules import crontab

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
# swivl/
APPS_DIR = BASE_DIR / "swivl"
env = environ.Env()

READ_DOT_ENV_FILE = env.bool("DJANGO_READ_DOT_ENV_FILE", default=True)
if READ_DOT_ENV_FILE:
    # OS environment variables take precedence over variables from .env
    env.read_env(str(BASE_DIR / ".env"))

# GENERAL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#debug
DEBUG = env.bool("DJANGO_DEBUG", False)
# Opt-in only: allows POST /api/onboarding/local-dev-billing-complete/ when DJANGO_DEBUG is False.
# Use on staging with Next.js NEXT_PUBLIC_SKIP_ONBOARDING_STRIPE=true. Never enable in production.
ALLOW_ONBOARDING_BILLING_BYPASS = env.bool("ALLOW_ONBOARDING_BILLING_BYPASS", default=False)
# Local time zone. Choices are
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# though not all of them may be available with every OS.
# In Windows, this must be set to your system time zone.
# Default is UTC; set DJANGO_TIME_ZONE=America/Denver for Mountain time in admin/UI.
TIME_ZONE = env("DJANGO_TIME_ZONE", default="UTC")
# https://docs.djangoproject.com/en/dev/ref/settings/#language-code
LANGUAGE_CODE = "en-us"
# https://docs.djangoproject.com/en/dev/ref/settings/#languages
# from django.utils.translation import gettext_lazy as _
# LANGUAGES = [
#     ('en', _('English')),
#     ('fr-fr', _('French')),
#     ('pt-br', _('Portuguese')),
# ]
# https://docs.djangoproject.com/en/dev/ref/settings/#site-id
SITE_ID = 1
# https://docs.djangoproject.com/en/dev/ref/settings/#use-i18n
USE_I18N = True
# https://docs.djangoproject.com/en/dev/ref/settings/#use-tz
USE_TZ = True
# https://docs.djangoproject.com/en/dev/ref/settings/#locale-paths
LOCALE_PATHS = [str(BASE_DIR / "locale")]

# DATABASES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#databases
DATABASES = {"default": env.db("DATABASE_URL")}
DATABASES["default"]["ATOMIC_REQUESTS"] = True
# https://docs.djangoproject.com/en/stable/ref/settings/#std:setting-DEFAULT_AUTO_FIELD
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# URLS
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#root-urlconf
ROOT_URLCONF = "config.urls"
# https://docs.djangoproject.com/en/dev/ref/settings/#wsgi-application
WSGI_APPLICATION = "config.wsgi.application"

# APPS
# ------------------------------------------------------------------------------
DJANGO_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.sites",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # "django.contrib.humanize", # Handy template tags
    "django.contrib.admin",
    "django.forms",
]
THIRD_PARTY_APPS = [
    "crispy_forms",
    "crispy_bootstrap5",
    "allauth",
    "allauth.account",
    "allauth.mfa",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "allauth.socialaccount.providers.microsoft",
    "rest_framework",
    "rest_framework.authtoken",
    "corsheaders",
    "drf_spectacular",
]

LOCAL_APPS = [
    "swivl.users",
    "accounts.apps.AccountsConfig",
    # Your stuff: custom apps go here
]
# https://docs.djangoproject.com/en/dev/ref/settings/#installed-apps
INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# MIGRATIONS
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#migration-modules
MIGRATION_MODULES = {"sites": "swivl.contrib.sites.migrations"}

# AUTHENTICATION
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#authentication-backends
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-user-model
AUTH_USER_MODEL = "users.User"
# https://docs.djangoproject.com/en/dev/ref/settings/#login-redirect-url
#LOGIN_REDIRECT_URL = "users:redirect"
# https://docs.djangoproject.com/en/dev/ref/settings/#login-url
LOGIN_URL = "account_login"

# PASSWORDS
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#password-hashers
PASSWORD_HASHERS = [
    # https://docs.djangoproject.com/en/dev/topics/auth/passwords/#using-argon2-with-django
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
]
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# MIDDLEWARE
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#middleware
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "accounts.middleware.RedirectNonStaffApiHostPagesMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

# STATIC
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#static-root
STATIC_ROOT = str(BASE_DIR / "staticfiles")
# https://docs.djangoproject.com/en/dev/ref/settings/#static-url
STATIC_URL = "/static/"
# https://docs.djangoproject.com/en/dev/ref/contrib/staticfiles/#std:setting-STATICFILES_DIRS
STATICFILES_DIRS = [str(APPS_DIR / "static")]
# https://docs.djangoproject.com/en/dev/ref/contrib/staticfiles/#staticfiles-finders
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]

# MEDIA
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#media-root
MEDIA_ROOT = str(APPS_DIR / "media")
# https://docs.djangoproject.com/en/dev/ref/settings/#media-url
MEDIA_URL = "/media/"

# TEMPLATES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#templates
TEMPLATES = [
    {
        # https://docs.djangoproject.com/en/dev/ref/settings/#std:setting-TEMPLATES-BACKEND
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # https://docs.djangoproject.com/en/dev/ref/settings/#dirs
        "DIRS": [str(APPS_DIR / "templates")],
        # https://docs.djangoproject.com/en/dev/ref/settings/#app-dirs
        "APP_DIRS": True,
        "OPTIONS": {
            # https://docs.djangoproject.com/en/dev/ref/settings/#template-context-processors
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.template.context_processors.i18n",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "django.template.context_processors.tz",
                "django.contrib.messages.context_processors.messages",
                "swivl.users.context_processors.allauth_settings",
            ],
        },
    },
]

# https://docs.djangoproject.com/en/dev/ref/settings/#form-renderer
FORM_RENDERER = "django.forms.renderers.TemplatesSetting"

# http://django-crispy-forms.readthedocs.io/en/latest/install.html#template-packs
CRISPY_TEMPLATE_PACK = "bootstrap5"
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"

# FIXTURES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#fixture-dirs
FIXTURE_DIRS = (str(APPS_DIR / "fixtures"),)

# SECURITY
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#session-cookie-httponly
SESSION_COOKIE_HTTPONLY = True
# https://docs.djangoproject.com/en/dev/ref/settings/#csrf-cookie-httponly
CSRF_COOKIE_HTTPONLY = True
# https://docs.djangoproject.com/en/dev/ref/settings/#x-frame-options
X_FRAME_OPTIONS = "DENY"

# EMAIL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#email-backend
EMAIL_BACKEND = env(
    "DJANGO_EMAIL_BACKEND",
    default="django.core.mail.backends.smtp.EmailBackend",
)
# https://docs.djangoproject.com/en/dev/ref/settings/#email-timeout
EMAIL_TIMEOUT = 5

# ADMIN
# ------------------------------------------------------------------------------
# Django Admin URL.
ADMIN_URL = "admin/"
# https://docs.djangoproject.com/en/dev/ref/settings/#admins
ADMINS = [("""Kevin Bacon""", "support@ripplerank.ai")]
# https://docs.djangoproject.com/en/dev/ref/settings/#managers
MANAGERS = ADMINS
# https://cookiecutter-django.readthedocs.io/en/latest/settings.html#other-environment-settings
# Force the `admin` sign in process to go through the `django-allauth` workflow
DJANGO_ADMIN_FORCE_ALLAUTH = env.bool("DJANGO_ADMIN_FORCE_ALLAUTH", default=False)

# LOGGING
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#logging
# See https://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}

REDIS_URL = env("REDIS_URL", default="redis://redis:6379/0")
REDIS_SSL = REDIS_URL.startswith("rediss://")

# Celery (async tasks; e.g. SEO enrichment)
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default=REDIS_URL)
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_BACKEND = None  # we don't need to store task results for SEO tasks
CELERY_TASK_IGNORE_RESULT = True
CELERY_TIMEZONE = TIME_ZONE
# Periodic full AEO monitoring re-run (optional). Task body no-ops unless enabled + profile IDs set.
# Env: AEO_SCHEDULED_FULL_MONITORING_ENABLED (default false), AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS
# (comma-separated ints; empty = no-op), AEO_SCHEDULED_FULL_MONITORING_CRON_HOUR / _CRON_MINUTE (UTC).
AEO_SCHEDULED_FULL_MONITORING_ENABLED = env.bool("AEO_SCHEDULED_FULL_MONITORING_ENABLED", default=False)
AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS = env.list("AEO_SCHEDULED_FULL_MONITORING_PROFILE_IDS", default=[])
AEO_SCHEDULED_FULL_MONITORING_CRON_HOUR = env.int("AEO_SCHEDULED_FULL_MONITORING_CRON_HOUR", default=4)
AEO_SCHEDULED_FULL_MONITORING_CRON_MINUTE = env.int("AEO_SCHEDULED_FULL_MONITORING_CRON_MINUTE", default=0)
CELERY_BEAT_SCHEDULE = {
    "aeo-scheduled-full-monitoring": {
        "task": "accounts.tasks.aeo_scheduled_full_monitoring_tick_task",
        "schedule": crontab(
            hour=AEO_SCHEDULED_FULL_MONITORING_CRON_HOUR,
            minute=AEO_SCHEDULED_FULL_MONITORING_CRON_MINUTE,
        ),
    },
}
# Optional split workers: celery -A config worker -Q celery,aeo_openai,aeo_gemini
# Phase-2 still runs in-process today; dedicated provider tasks can use these via apply_async(queue=...).
AEO_OPENAI_CELERY_QUEUE = env("AEO_OPENAI_CELERY_QUEUE", default="celery")
AEO_GEMINI_CELERY_QUEUE = env("AEO_GEMINI_CELERY_QUEUE", default="celery")
AEO_PROVIDER_HTTP_MAX_RETRIES = env.int("AEO_PROVIDER_HTTP_MAX_RETRIES", default=5)


# django-allauth
# ------------------------------------------------------------------------------
ACCOUNT_ALLOW_REGISTRATION = env.bool("DJANGO_ACCOUNT_ALLOW_REGISTRATION", True)
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_LOGIN_METHODS = {"username"}
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*", "password1*", "password2*"]
# https://docs.allauth.org/en/latest/account/configuration.html
# ``mandatory`` sends users to /accounts/confirm-email/ until verified; we have no outbound
# email yet. Use ``optional`` so login succeeds; set ``DJANGO_ACCOUNT_EMAIL_VERIFICATION=mandatory``
# when SMTP + verification emails are configured.
ACCOUNT_EMAIL_VERIFICATION = env("DJANGO_ACCOUNT_EMAIL_VERIFICATION", default="optional")
# Google (and similar) already verify email; do not block OAuth users on Django email confirmation.
# Without this, social login often redirects to /accounts/confirm-email/ instead of the SPA.
SOCIALACCOUNT_EMAIL_VERIFICATION = "none"
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_ADAPTER = "swivl.users.adapters.AccountAdapter"
# https://docs.allauth.org/en/latest/account/forms.html
ACCOUNT_FORMS = {"signup": "swivl.users.forms.UserSignupForm"}
# https://docs.allauth.org/en/latest/socialaccount/configuration.html
SOCIALACCOUNT_ADAPTER = "swivl.users.adapters.SocialAccountAdapter"
# https://docs.allauth.org/en/latest/socialaccount/configuration.html
SOCIALACCOUNT_FORMS = {"signup": "swivl.users.forms.UserSocialSignupForm"}
SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "APP": {
            "client_id": env("GOOGLE_CLIENT_ID", default="your-google-client-id"),
            "secret": env(
                "GOOGLE_CLIENT_SECRET",
                default="your-google-client-secret",
            ),
            "key": "",
        },
        "SCOPE": ["email", "profile"],
        "AUTH_PARAMS": {"access_type": "online"},
    },
    "microsoft": {
        "APP": {
            "client_id": env("MICROSOFT_CLIENT_ID", default=""),
            "secret": env("MICROSOFT_CLIENT_SECRET", default=""),
            "key": "",
        },
        "SCOPE": ["openid", "email", "profile", "User.Read"],
        "AUTH_PARAMS": {"prompt": "select_account"},
        "TENANT": env("MICROSOFT_TENANT", default="common"),
    },
}

# django-rest-framework
# -------------------------------------------------------------------------------
# django-rest-framework - https://www.django-rest-framework.org/api-guide/settings/
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# django-cors-headers - https://github.com/adamchainz/django-cors-headers#setup
CORS_URLS_REGEX = r"^/api/.*$"
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]
CORS_ALLOW_CREDENTIALS = True

# CSRF: trust calls coming from the frontend origin
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
]

# By Default swagger ui is available only to admin user(s). You can change permission classes to change that
# See more configuration options at https://drf-spectacular.readthedocs.io/en/latest/settings.html#settings
SPECTACULAR_SETTINGS = {
    "TITLE": "Swivl API",
    "DESCRIPTION": "Documentation of API endpoints of Swivl",
    "VERSION": "1.0.0",
    "SERVE_PERMISSIONS": ["rest_framework.permissions.IsAdminUser"],
    "SCHEMA_PATH_PREFIX": "/api/",
}

# Production: https://app.ripplerank.ai (see config.settings.production for defaults).
FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL",
    "http://localhost:3000",
).rstrip("/")
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
LOGIN_REDIRECT_URL = f"{FRONTEND_BASE_URL}/onboarding"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "123")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/accounts/google/login/callback/")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "123")
MICROSOFT_CLIENT_ID = os.environ.get("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.environ.get("MICROSOFT_CLIENT_SECRET", "")
MICROSOFT_TENANT = os.environ.get("MICROSOFT_TENANT", "common")
GOOGLE_ADS_DEVELOPER_TOKEN = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "123")
GOOGLE_ADS_CUSTOMER_ID = os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "123")
DATAFORSEO_LOGIN = os.environ.get("DATAFORSEO_LOGIN", "123")
DATAFORSEO_PASSWORD = os.environ.get("DATAFORSEO_PASSWORD", "123")
DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS = env.bool(
    "DATAFORSEO_DISABLE_COMPETITOR_LOOKUPS",
    default=False,
)
# Labs ranked_keywords/live: max items per request (API ceiling varies; default 100).
DATAFORSEO_RANKED_KEYWORDS_LIMIT = env.int("DATAFORSEO_RANKED_KEYWORDS_LIMIT", default=100)
# Cap rows stored on SEOOverviewSnapshot.top_keywords (ranked + gap + profile seeds). Pagination is client-side.
SEO_TOP_KEYWORDS_MAX_PERSISTED = env.int("SEO_TOP_KEYWORDS_MAX_PERSISTED", default=200)
AEO_TESTING_MODE = env.bool("AEO_TESTING_MODE", default=False)
AEO_TEST_PROMPT_COUNT = env.int("AEO_TEST_PROMPT_COUNT", default=10)
AEO_PROD_PROMPT_COUNT = env.int("AEO_PROD_PROMPT_COUNT", default=50)
# OpenAI prompt-batch generation: output max_tokens by combined target (starter ≤10, pro ≤50, else advanced).
# Larger values reduce truncated JSON / failed_empty for long prompt lists; TPM/RPM are account-level limits.
AEO_OPENAI_MAX_TOKENS_STARTER = env.int("AEO_OPENAI_MAX_TOKENS_STARTER", default=1024)
AEO_OPENAI_MAX_TOKENS_PRO = env.int("AEO_OPENAI_MAX_TOKENS_PRO", default=4096)
AEO_OPENAI_MAX_TOKENS_ADVANCED = env.int("AEO_OPENAI_MAX_TOKENS_ADVANCED", default=8192)
# Shortfall top-up after the four-type pass (build_full_aeo_prompt_plan): extra buffer on requested batch size for dedupe loss.
AEO_PROMPT_TOPUP_MAX_ROUNDS = env.int("AEO_PROMPT_TOPUP_MAX_ROUNDS", default=3)
AEO_PROMPT_TOPUP_BUFFER = env.int("AEO_PROMPT_TOPUP_BUFFER", default=8)
AEO_ENABLE_RECOMMENDATION_STAGE = env.bool("AEO_ENABLE_RECOMMENDATION_STAGE", default=False)
AEO_FULL_PHASE_ETA_K = env.int("AEO_FULL_PHASE_ETA_K", default=5)
AEO_FULL_PHASE_ETA_DEFAULT_SEC = env.int("AEO_FULL_PHASE_ETA_DEFAULT_SEC", default=120)
AEO_FULL_PHASE_ETA_CAP_SEC = env.int("AEO_FULL_PHASE_ETA_CAP_SEC", default=3600)
# Phase 5 recommendation nl_explanation: False = template-only (no OpenAI); True = LLM (extra API calls).
AEO_RECOMMENDATION_USE_OPENAI = env.bool("AEO_RECOMMENDATION_USE_OPENAI", default=False)
# Max actionable recommendations per run (3–5 typical). Env override for ops rollback.
AEO_RECOMMENDATION_MAX_LEAVES = env.int("AEO_RECOMMENDATION_MAX_LEAVES", default=5)
# Cluster visibility/citation gaps by (absence_reason, content_angle, action_type) before capping.
AEO_RECOMMENDATION_GROUP_GAPS = env.bool("AEO_RECOMMENDATION_GROUP_GAPS", default=True)
# Google Gemini — optional Phase 2 AEO execution alongside OpenAI (see accounts.gemini_utils).
# Also accepted: GOOGLE_GEMINI_API_KEY if this is empty (read in gemini_utils).
GEMINI_API_KEY = env("GEMINI_API_KEY", default="").strip()
AEO_GEMINI_EXECUTION_MODEL = env("AEO_GEMINI_EXECUTION_MODEL", default="gemini-2.5-flash")
# Gemini review topics: only used when ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK is True (default off).
# Empty → use AEO_GEMINI_EXECUTION_MODEL (see accounts.onboarding_review_topics.get_gemini_review_topics_model).
GEMINI_REVIEW_TOPICS_MODEL = env("GEMINI_REVIEW_TOPICS_MODEL", default="").strip()
# Perplexity Sonar — optional Phase 2 AEO execution alongside OpenAI/Gemini (see accounts.aeo.perplexity_execution_utils).
PERPLEXITY_API_KEY = env("PERPLEXITY_API_KEY", default="").strip()
PERPLEXITY_AEO_MODEL = env("PERPLEXITY_AEO_MODEL", default="sonar").strip() or "sonar"
# Onboarding crawl ``review_topics`` (domain-only). Empty → same as PERPLEXITY_AEO_MODEL.
PERPLEXITY_ONBOARDING_REVIEW_TOPICS_MODEL = env("PERPLEXITY_ONBOARDING_REVIEW_TOPICS_MODEL", default="").strip()
# When False (default), onboarding review topics never call Gemini; set PERPLEXITY_API_KEY or enable fallback.
ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK = env.bool(
    "ONBOARDING_REVIEW_TOPICS_USE_GEMINI_FALLBACK", default=False
)
# Stripe billing / webhook configuration.
STRIPE_SECRET_KEY = env("STRIPE_SECRET_KEY", default="").strip()
STRIPE_WEBHOOK_SECRET = env("STRIPE_WEBHOOK_SECRET", default="").strip()
# Price IDs on subscription/invoice line items → mapped to BusinessProfile.plan in accounts.stripe_billing.
STRIPE_PRICE_ID_STARTER_MONTHLY = env("STRIPE_PRICE_ID_STARTER_MONTHLY", default="").strip()
STRIPE_PRICE_ID_STARTER_YEARLY = env("STRIPE_PRICE_ID_STARTER_YEARLY", default="").strip()
STRIPE_PRICE_ID_PRO_MONTHLY = env("STRIPE_PRICE_ID_PRO_MONTHLY", default="").strip()
STRIPE_PRICE_ID_PRO_YEARLY = env("STRIPE_PRICE_ID_PRO_YEARLY", default="").strip()
STRIPE_PRICE_ID_ADVANCED_MONTHLY = env("STRIPE_PRICE_ID_ADVANCED_MONTHLY", default="").strip()
STRIPE_PRICE_ID_ADVANCED_YEARLY = env("STRIPE_PRICE_ID_ADVANCED_YEARLY", default="").strip()
# Payment Link IDs or buy URLs — must match live links used at checkout so checkout.session.completed
# can resolve tier when the session has no price/line_items (see plan_mapping_by_payment_link_id).
STRIPE_PAYMENT_LINK_STARTER_MONTHLY = env("STRIPE_PAYMENT_LINK_STARTER_MONTHLY", default="").strip()
STRIPE_PAYMENT_LINK_STARTER_YEARLY = env("STRIPE_PAYMENT_LINK_STARTER_YEARLY", default="").strip()
STRIPE_PAYMENT_LINK_PRO_MONTHLY = env("STRIPE_PAYMENT_LINK_PRO_MONTHLY", default="").strip()
STRIPE_PAYMENT_LINK_PRO_YEARLY = env("STRIPE_PAYMENT_LINK_PRO_YEARLY", default="").strip()
STRIPE_PAYMENT_LINK_ADVANCED_MONTHLY = env("STRIPE_PAYMENT_LINK_ADVANCED_MONTHLY", default="").strip()
STRIPE_PAYMENT_LINK_ADVANCED_YEARLY = env("STRIPE_PAYMENT_LINK_ADVANCED_YEARLY", default="").strip()
# Meta (Facebook) – Ads app: OAuth login + Marketing API (connect button and ad campaigns)
META_ADS_APP_ID = os.environ.get("META_ADS_APP_ID", "")
META_ADS_APP_SECRET = os.environ.get("META_ADS_APP_SECRET", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Chat completions model for SEO/Reviews agents, onboarding, and AEO (unless overridden below).
OPENAI_MODEL = env("OPENAI_MODEL", default="gpt-5-mini").strip() or "gpt-4o-mini"
# Optional: different model for AEO Phase 2 execution / structured extraction (defaults to OPENAI_MODEL).
AEO_EXECUTION_MODEL = env("AEO_EXECUTION_MODEL", default=OPENAI_MODEL).strip() or OPENAI_MODEL
# Max concurrent threads for AEO batch execution (OpenAI/Gemini jobs); clamped in code to 1–64.
AEO_EXECUTION_MAX_WORKERS = env.int("AEO_EXECUTION_MAX_WORKERS", default=20)
AEO_EXTRACTION_PARSER_MODEL = env("AEO_EXTRACTION_PARSER_MODEL", default=OPENAI_MODEL).strip() or OPENAI_MODEL
# Phase 3 wrong-URL verification (DNS/HTTP probe of model-attributed non-canonical links).
AEO_DOMAIN_VERIFY_ENABLED = env.bool("AEO_DOMAIN_VERIFY_ENABLED", default=True)
AEO_DOMAIN_VERIFY_TIMEOUT_S = env.float("AEO_DOMAIN_VERIFY_TIMEOUT_S", default=3.0)
AEO_DOMAIN_VERIFY_MAX_REDIRECTS = env.int("AEO_DOMAIN_VERIFY_MAX_REDIRECTS", default=5)
AEO_DOMAIN_VERIFY_USER_AGENT = env("AEO_DOMAIN_VERIFY_USER_AGENT", default="").strip()
# Lowercase hostnames that bypass private-IP blocking (unit tests only; do not use in production).
AEO_DOMAIN_VERIFY_ALLOWLIST = env.list("AEO_DOMAIN_VERIFY_ALLOWLIST", default=[])