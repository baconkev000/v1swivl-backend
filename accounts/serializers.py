from django.contrib.auth import get_user_model
from rest_framework import serializers
import logging

from .models import BusinessProfile, SEOOverviewSnapshot
from .dataforseo_utils import get_or_refresh_seo_score_for_user


logger = logging.getLogger(__name__)
User = get_user_model()


class BusinessProfileSerializer(serializers.ModelSerializer):
    website_url = serializers.CharField(
        required=False,
        allow_blank=True,
    )
    email = serializers.EmailField(source="user.email", read_only=True)
    seo_score = serializers.SerializerMethodField()
    search_performance_score = serializers.SerializerMethodField()
    onpage_seo_score = serializers.SerializerMethodField()
    technical_seo_score = serializers.SerializerMethodField()
    pages_audited = serializers.SerializerMethodField()
    onpage_issue_summaries = serializers.SerializerMethodField()
    search_visibility_percent = serializers.SerializerMethodField()
    missed_searches_monthly = serializers.SerializerMethodField()
    total_search_volume = serializers.SerializerMethodField()
    organic_visitors = serializers.SerializerMethodField()
    top_keywords = serializers.SerializerMethodField()
    seo_next_steps = serializers.SerializerMethodField()

    class Meta:
        model = BusinessProfile
        fields = [
            "id",
            "email",
            "full_name",
            "business_name",
            "business_address",
            "industry",
            "tone_of_voice",
            "phone",
            "description",
            "website_url",
            "plan",
            "seo_score",
            "search_performance_score",
            "onpage_seo_score",
            "technical_seo_score",
            "pages_audited",
            "onpage_issue_summaries",
            "search_visibility_percent",
            "missed_searches_monthly",
            "total_search_volume",
            "organic_visitors",
            "top_keywords",
            "seo_next_steps",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "email", "created_at", "updated_at"]

    def validate_website_url(self, value):
        """Normalize URL to include scheme."""
        if value:
            value = value.strip()
            if not value.startswith(("http://", "https://")):
                value = "https://" + value
        return value

    def _get_seo_bundle(self, obj: BusinessProfile) -> dict | None:
        user = getattr(obj, "user", None)
        site_url = obj.website_url or ""

        if not user:
            logger.warning(
                "[BusinessProfileSerializer] get_seo_score: missing user for profile id=%s",
                getattr(obj, "id", None),
            )
            return None

        # Simple per-instance cache so we only call the helper once per profile.
        cache_attr = "_seo_bundle_cache"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        try:
            logger.info(
                "[BusinessProfileSerializer] get_seo_score: user_id=%s site_url=%s",
                getattr(user, "id", None),
                site_url,
            )
            data = get_or_refresh_seo_score_for_user(user, site_url=site_url or None)
            if not data:
                logger.info(
                    "[BusinessProfileSerializer] get_seo_score: no data returned for user_id=%s",
                    getattr(user, "id", None),
                )
                setattr(self, cache_attr, None)
                return None

            logger.info(
                "[BusinessProfileSerializer] get_seo_score: resolved seo_bundle=%s for user_id=%s",
                {k: data.get(k) for k in ["seo_score", "search_performance_score", "onpage_seo_score", "technical_seo_score", "pages_audited"]},
                getattr(user, "id", None),
            )
            setattr(self, cache_attr, data)
            return data
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[BusinessProfileSerializer] get_seo_score: exception for user_id=%s: %s",
                getattr(user, "id", None),
                str(exc)[:300],
            )
            setattr(self, cache_attr, None)
            return None

    def get_seo_score(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        score = bundle.get("seo_score")
        return int(score) if score is not None else None

    def get_search_performance_score(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        score = bundle.get("search_performance_score")
        return int(score) if score is not None else None

    def get_onpage_seo_score(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        score = bundle.get("onpage_seo_score")
        return int(score) if score is not None else None

    def get_technical_seo_score(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        score = bundle.get("technical_seo_score")
        return int(score) if score is not None else None

    def get_pages_audited(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("pages_audited")
        return int(val) if val is not None else None

    def get_onpage_issue_summaries(self, obj: BusinessProfile) -> dict | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        return bundle.get("onpage_issue_summaries") or {}

    def get_search_visibility_percent(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("search_visibility_percent")
        return int(val) if val is not None else None

    def get_missed_searches_monthly(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("missed_searches_monthly")
        return int(val) if val is not None else None

    def get_total_search_volume(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("total_search_volume")
        return int(val) if val is not None else None

    def get_organic_visitors(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("organic_visitors")
        return int(val) if val is not None else None

    def get_top_keywords(self, obj: BusinessProfile):
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return []
        return bundle.get("top_keywords") or []

    def get_seo_next_steps(self, obj: BusinessProfile):
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return []
        return bundle.get("seo_next_steps") or []
