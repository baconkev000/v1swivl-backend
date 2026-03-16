# ruff: noqa: ERA001, E501
"""Base settings to build other settings files upon."""

from pathlib import Path
import os
import environ

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
# swivl/
APPS_DIR = BASE_DIR / "swivl"
env = environ.Env()

READ_DOT_ENV_FILE = env.bool("DJANGO_READ_DOT_ENV_FILE", default=False)
if READ_DOT_ENV_FILE:
    # OS environment variables take precedence over variables from .env
    env.read_env(str(BASE_DIR / ".env"))

# GENERAL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#debug
DEBUG = env.bool("DJANGO_DEBUG", False)
# Local time zone. Choices are
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# though not all of them may be available with every OS.
# In Windows, this must be set to your system time zone.
TIME_ZONE = "UTC"
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
    "rest_framework",
    "rest_framework.authtoken",
    "corsheaders",
    "drf_spectacular",
]

LOCAL_APPS = [
    "swivl.users",
    "accounts",
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
ADMINS = [("""Kevin Bacon""", "support@getswivl.ai")]
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


# django-allauth
# ------------------------------------------------------------------------------
ACCOUNT_ALLOW_REGISTRATION = env.bool("DJANGO_ACCOUNT_ALLOW_REGISTRATION", True)
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_LOGIN_METHODS = {"username"}
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*", "password1*", "password2*"]
# https://docs.allauth.org/en/latest/account/configuration.html
ACCOUNT_EMAIL_VERIFICATION = "mandatory"
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

FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL",
    "http://localhost:3000"
)
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
# settings.py
LOGIN_REDIRECT_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:3000") + "/app"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "123")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/accounts/google/login/callback/")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "123")
GOOGLE_ADS_DEVELOPER_TOKEN = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "123")
GOOGLE_ADS_CUSTOMER_ID = os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "123")
DATAFORSEO_LOGIN = os.environ.get("DATAFORSEO_LOGIN", "123")
DATAFORSEO_PASSWORD = os.environ.get("DATAFORSEO_PASSWORD", "123")
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:3000")

# Meta (Facebook) – Ads app: OAuth login + Marketing API (connect button and ad campaigns)
META_ADS_APP_ID = os.environ.get("META_ADS_APP_ID", "")
META_ADS_APP_SECRET = os.environ.get("META_ADS_APP_SECRET", "")