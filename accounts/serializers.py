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

    def get_seo_score(self, obj: BusinessProfile) -> int | None:
        """
        Return the latest SEO score for this business profile.

        Logging is added to help debug any issues fetching or computing the score.
        """
        user = getattr(obj, "user", None)
        site_url = obj.website_url or ""

        if not user:
            logger.warning(
                "[BusinessProfileSerializer] get_seo_score: missing user for profile id=%s",
                getattr(obj, "id", None),
            )
            return None

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
                return None

            score = data.get("seo_score")
            logger.info(
                "[BusinessProfileSerializer] get_seo_score: resolved seo_score=%s for user_id=%s",
                score,
                getattr(user, "id", None),
            )
            return int(score) if score is not None else None
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[BusinessProfileSerializer] get_seo_score: exception for user_id=%s: %s",
                getattr(user, "id", None),
                str(exc)[:300],
            )
            return None
