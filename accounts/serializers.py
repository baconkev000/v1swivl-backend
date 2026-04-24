import logging
from urllib.parse import urlparse

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import serializers

from .constants import AEO_RECOMMENDATIONS_TTL, AEO_SNAPSHOT_TTL, SEO_SNAPSHOT_TTL
from .domain_utils import normalize_tracked_competitor_domain
from .models import BusinessProfile, SEOOverviewSnapshot, AEOOverviewSnapshot, TrackedCompetitor
from .aeo.aeo_plan_targets import (
    aeo_effective_custom_prompt_cap_for_profile,
    aeo_effective_custom_prompt_cap_for_validation,
    aeo_effective_cap_for_validation,
    aeo_effective_monitored_target_for_profile,
    aeo_effective_total_selected_slots_for_validation,
    aeo_fallback_global_target_count,
    aeo_onboarding_min_for_validation,
)
from .dataforseo_utils import get_or_refresh_seo_score_for_user
from .dataforseo_utils import (
    get_aeo_content_readiness_for_site,
    get_profile_location_code,
    normalize_domain,
    seo_snapshot_context_for_profile,
)
from .openai_utils import generate_aeo_recommendations
from .onboarding_completion import business_profile_fully_onboarded
from . import debug_log as _debug


logger = logging.getLogger(__name__)
User = get_user_model()


def _normalize_stored_website_url(value: str) -> str | None:
    """
    Reduce a user-entered URL to scheme + host for storage (BusinessProfile.website_url is max 200).

    Strips path, query string, fragment, and userinfo. Drops leading www. on the host.
    Returns None if no usable host can be parsed (caller may treat as validation error).
    """
    v = (value or "").strip()
    if not v:
        return ""
    if not v.startswith(("http://", "https://")):
        v = "https://" + v
    parsed = urlparse(v)
    netloc = (parsed.netloc or "").strip().lower()
    if not netloc:
        path_part = (parsed.path or "").lstrip("/")
        first = path_part.split("/")[0] if path_part else ""
        if first and ".." not in first and "/" not in first and not first.startswith("."):
            netloc = first.lower()
        else:
            return None
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if not netloc:
        return None
    scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"
    return f"{scheme}://{netloc}"


def _fallback_top_keywords_from_stored_snapshots(profile: BusinessProfile) -> list:
    """
    When get_or_refresh_seo_score_for_user returns no keywords (TTL, Labs failure, etc.),
    still surface the latest non-empty top_keywords saved on SEOOverviewSnapshot for this
    user, domain, and location scope (same keys as snapshot pipeline).
    """
    site = str(getattr(profile, "website_url", "") or "").strip()
    domain = normalize_domain(site) if site else None
    if not domain:
        return []
    loc_mode, loc_code = seo_snapshot_context_for_profile(profile)
    qs = (
        SEOOverviewSnapshot.objects.filter(
            business_profile=profile,
            cached_domain__iexact=domain,
            cached_location_mode=str(loc_mode or "organic"),
            cached_location_code=int(loc_code or 0),
        )
        .order_by("-last_fetched_at", "-id")
        .only("top_keywords")
    )
    for snap in qs.iterator(chunk_size=32):
        tk = getattr(snap, "top_keywords", None) or []
        if isinstance(tk, list) and len(tk) > 0:
            return list(tk)
    return []


def _skip_heavy_latest_seo_snapshot_actions_payload(profile: BusinessProfile) -> dict:
    """
    When ``skip_heavy_profile_metrics`` avoids calling DataForSEO, still expose stored
    SEO overview actions from the latest ``SEOOverviewSnapshot`` for this business profile
    (same ordering as ``get_top_keywords`` skip-heavy branch: ``-last_fetched_at``, ``-id``).
    """
    if not profile:
        return {
            "seo_next_steps": [],
            "keyword_action_suggestions": [],
            "seo_structured_issues": [],
            "enrichment_status": "complete",
        }
    snap = (
        SEOOverviewSnapshot.objects.filter(business_profile=profile)
        .order_by("-last_fetched_at", "-id")
        .only(
            "seo_next_steps",
            "keyword_action_suggestions",
            "seo_structured_issues",
            "keywords_enriched_at",
            "seo_next_steps_refreshed_at",
            "keyword_action_suggestions_refreshed_at",
        )
        .first()
    )
    if snap is None:
        return {
            "seo_next_steps": [],
            "keyword_action_suggestions": [],
            "seo_structured_issues": [],
            "enrichment_status": "complete",
        }
    steps = getattr(snap, "seo_next_steps", None) or []
    kws = getattr(snap, "keyword_action_suggestions", None) or []
    issues = getattr(snap, "seo_structured_issues", None) or []
    enrichment_status = "complete" if (
        getattr(snap, "keywords_enriched_at", None)
        and getattr(snap, "seo_next_steps_refreshed_at", None)
        and getattr(snap, "keyword_action_suggestions_refreshed_at", None)
    ) else "pending"
    return {
        "seo_next_steps": list(steps) if isinstance(steps, list) else [],
        "keyword_action_suggestions": list(kws) if isinstance(kws, list) else [],
        "seo_structured_issues": list(issues) if isinstance(issues, list) else [],
        "enrichment_status": enrichment_status,
    }


def _aeo_prompt_target_count() -> int:
    """Backward-compatible global default when no profile is in context (tests, helpers)."""
    return aeo_fallback_global_target_count()


class BusinessProfileSerializer(serializers.ModelSerializer):
    website_url = serializers.CharField(
        required=False,
        allow_blank=True,
    )
    customer_reach = serializers.ChoiceField(
        choices=[
            BusinessProfile.CUSTOMER_REACH_ONLINE,
            BusinessProfile.CUSTOMER_REACH_LOCAL,
        ],
        required=False,
    )
    customer_reach_state = serializers.CharField(required=False, allow_blank=True)
    customer_reach_city = serializers.CharField(required=False, allow_blank=True)
    email = serializers.EmailField(source="user.email", required=False)
    plan = serializers.ChoiceField(
        choices=[
            BusinessProfile.PLAN_NONE,
            BusinessProfile.PLAN_STARTER,
            BusinessProfile.PLAN_PRO,
            BusinessProfile.PLAN_ADVANCED,
        ],
        required=False,
        allow_blank=True,
    )
    tracked_competitors = serializers.SerializerMethodField()
    seo_score = serializers.SerializerMethodField()
    search_performance_score = serializers.SerializerMethodField()
    search_visibility_percent = serializers.SerializerMethodField()
    missed_searches_monthly = serializers.SerializerMethodField()
    total_search_volume = serializers.SerializerMethodField()
    estimated_search_appearances_monthly = serializers.SerializerMethodField()
    organic_visitors = serializers.SerializerMethodField()
    top_keywords = serializers.SerializerMethodField()
    seo_next_steps = serializers.SerializerMethodField()
    aeo_score = serializers.SerializerMethodField()
    question_coverage_score = serializers.SerializerMethodField()
    questions_found = serializers.SerializerMethodField()
    questions_missing = serializers.SerializerMethodField()
    faq_readiness_score = serializers.SerializerMethodField()
    faq_blocks_found = serializers.SerializerMethodField()
    faq_schema_present = serializers.SerializerMethodField()
    snippet_readiness_score = serializers.SerializerMethodField()
    answer_blocks_found = serializers.SerializerMethodField()
    aeo_recommendations = serializers.SerializerMethodField()
    aeo_status = serializers.SerializerMethodField()
    aeo_last_computed_at = serializers.SerializerMethodField()
    keyword_action_suggestions = serializers.SerializerMethodField()
    seo_structured_issues = serializers.SerializerMethodField()
    enrichment_status = serializers.SerializerMethodField()
    seo_metrics_location_mode = serializers.SerializerMethodField()
    seo_location_label = serializers.SerializerMethodField()
    local_verified_keyword_count = serializers.SerializerMethodField()
    local_verification_affects_visibility = serializers.SerializerMethodField()
    seo_competitor_domains_override = serializers.CharField(
        required=False,
        allow_blank=True,
    )
    selected_aeo_prompts = serializers.ListField(
        child=serializers.JSONField(required=False),
        required=False,
        allow_empty=True,
    )
    aeo_onboarding_prompt_target_count = serializers.SerializerMethodField()
    aeo_custom_prompt_cap = serializers.SerializerMethodField()
    viewer_team_role = serializers.SerializerMethodField()
    viewer_is_main_account_owner = serializers.SerializerMethodField()
    viewer_can_edit_company_profile = serializers.SerializerMethodField()
    viewer_can_access_billing = serializers.SerializerMethodField()
    viewer_can_manage_team = serializers.SerializerMethodField()
    viewer_email = serializers.SerializerMethodField()
    onboarding_complete = serializers.SerializerMethodField()

    class Meta:
        model = BusinessProfile
        fields = [
            "id",
            "email",
            "full_name",
            "business_name",
            "business_address",
            "customer_reach",
            "customer_reach_state",
            "customer_reach_city",
            "industry",
            "phone",
            "description",
            "website_url",
            "selected_aeo_prompts",
            "aeo_onboarding_prompt_target_count",
            "aeo_custom_prompt_cap",
            "aeo_prompt_expansion_status",
            "aeo_prompt_expansion_target",
            "aeo_prompt_expansion_progress",
            "aeo_prompt_expansion_last_error",
            "aeo_prompt_expansion_updated_at",
            "plan",
            "stripe_customer_id",
            "stripe_subscription_id",
            "stripe_price_id",
            "stripe_subscription_status",
            "stripe_current_period_end",
            "stripe_cancel_at_period_end",
            "is_main",
            "tracked_competitors",
            "seo_competitor_domains_override",
            "seo_location_mode",
            "seo_score",
            "search_performance_score",
            "search_visibility_percent",
            "missed_searches_monthly",
            "total_search_volume",
            "estimated_search_appearances_monthly",
            "organic_visitors",
            "top_keywords",
            "seo_next_steps",
            "aeo_score",
            "question_coverage_score",
            "questions_found",
            "questions_missing",
            "faq_readiness_score",
            "faq_blocks_found",
            "faq_schema_present",
            "snippet_readiness_score",
            "answer_blocks_found",
            "aeo_recommendations",
            "aeo_status",
            "aeo_last_computed_at",
            "keyword_action_suggestions",
            "seo_structured_issues",
            "enrichment_status",
            "seo_metrics_location_mode",
            "seo_location_label",
            "local_verified_keyword_count",
            "local_verification_affects_visibility",
            "created_at",
            "updated_at",
            "viewer_team_role",
            "viewer_is_main_account_owner",
            "viewer_can_edit_company_profile",
            "viewer_can_access_billing",
            "viewer_can_manage_team",
            "viewer_email",
            "onboarding_complete",
        ]
        read_only_fields = [
            "id",
            "email",
            "aeo_onboarding_prompt_target_count",
            "aeo_custom_prompt_cap",
            "aeo_prompt_expansion_status",
            "aeo_prompt_expansion_target",
            "aeo_prompt_expansion_progress",
            "aeo_prompt_expansion_last_error",
            "aeo_prompt_expansion_updated_at",
            "stripe_customer_id",
            "stripe_subscription_id",
            "stripe_price_id",
            "stripe_subscription_status",
            "stripe_current_period_end",
            "stripe_cancel_at_period_end",
            "created_at",
            "updated_at",
            "viewer_team_role",
            "viewer_is_main_account_owner",
            "viewer_can_edit_company_profile",
            "viewer_can_access_billing",
            "viewer_can_manage_team",
            "viewer_email",
            "onboarding_complete",
        ]

    def _viewer_access_dict(self, obj: BusinessProfile | None = None) -> dict:
        """
        Resolve viewer flags for the row being serialized.

        With ``many=True``, ``self.instance`` may be a queryset/list — always prefer the
        ``obj`` passed into SerializerMethodField getters.
        """
        profile: BusinessProfile | None = obj if isinstance(obj, BusinessProfile) else None
        if profile is None:
            inst = getattr(self, "instance", None)
            if isinstance(inst, BusinessProfile):
                profile = inst

        resolver = self.context.get("viewer_access_resolver")
        if callable(resolver) and profile is not None:
            resolved = resolver(profile)
            if isinstance(resolved, dict) and resolved:
                return resolved
        raw = self.context.get("viewer_access")
        if isinstance(raw, dict) and raw:
            return raw
        return {
            "viewer_team_role": "owner",
            "viewer_is_main_account_owner": True,
            "viewer_can_edit_company_profile": True,
            "viewer_can_access_billing": True,
            "viewer_can_manage_team": True,
        }

    def get_viewer_team_role(self, obj: BusinessProfile) -> str:
        return str(self._viewer_access_dict(obj).get("viewer_team_role") or "owner")

    def get_viewer_is_main_account_owner(self, obj: BusinessProfile) -> bool:
        return bool(self._viewer_access_dict(obj).get("viewer_is_main_account_owner"))

    def get_viewer_can_edit_company_profile(self, obj: BusinessProfile) -> bool:
        return bool(self._viewer_access_dict(obj).get("viewer_can_edit_company_profile", True))

    def get_viewer_can_access_billing(self, obj: BusinessProfile) -> bool:
        return bool(self._viewer_access_dict(obj).get("viewer_can_access_billing", True))

    def get_viewer_can_manage_team(self, obj: BusinessProfile) -> bool:
        return bool(self._viewer_access_dict(obj).get("viewer_can_manage_team", True))

    def get_viewer_email(self, obj: BusinessProfile) -> str:
        req = self.context.get("request")
        u = getattr(req, "user", None) if req is not None else None
        if u is not None and getattr(u, "is_authenticated", False) and getattr(u, "email", None):
            return str(u.email)
        return str(getattr(obj.user, "email", "") or "")

    def get_onboarding_complete(self, obj: BusinessProfile) -> bool:
        try:
            return bool(business_profile_fully_onboarded(obj))
        except Exception:
            return False

    def get_aeo_onboarding_prompt_target_count(self, obj: BusinessProfile) -> int:
        return aeo_effective_monitored_target_for_profile(obj)

    def get_aeo_custom_prompt_cap(self, obj: BusinessProfile) -> int:
        return aeo_effective_custom_prompt_cap_for_profile(obj)

    def get_tracked_competitors(self, obj: BusinessProfile) -> list[dict]:
        return [
            {"id": c.id, "name": c.name, "domain": c.domain}
            for c in obj.tracked_competitors.order_by("domain").all()
        ]

    @staticmethod
    def _normalize_tracked_competitors_input(raw: list) -> list[dict[str, str]]:
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise serializers.ValidationError(
                    {"tracked_competitors": f"Item {i} must be an object with name and domain."}
                )
            name = str(item.get("name") or "").strip()
            raw_dom = str(item.get("domain") or "").strip()
            nd = normalize_tracked_competitor_domain(raw_dom)
            if not nd:
                raise serializers.ValidationError(
                    {"tracked_competitors": f"Invalid domain at index {i}: {raw_dom!r}."}
                )
            if not name:
                name = nd
            if nd in seen:
                continue
            seen.add(nd)
            out.append({"name": name, "domain": nd})
        return out

    @staticmethod
    def _apply_tracked_competitors(instance: BusinessProfile, items: list[dict[str, str]]) -> None:
        ids: list[int] = []
        for item in items:
            obj, _created = TrackedCompetitor.objects.get_or_create(
                domain=item["domain"],
                defaults={"name": item["name"]},
            )
            if obj.name != item["name"]:
                obj.name = item["name"]
                obj.save(update_fields=["name", "updated_at"])
            ids.append(obj.id)
        instance.tracked_competitors.set(ids)

    def validate_website_url(self, value):
        """Normalize to origin (scheme + host only) so paths/query strings do not exceed URLField(200)."""
        if value is None or str(value).strip() == "":
            return ""
        normalized = _normalize_stored_website_url(str(value))
        if normalized is None:
            raise serializers.ValidationError("Enter a valid website URL.")
        if len(normalized) > 200:
            raise serializers.ValidationError("Website URL exceeds maximum length.")
        return normalized

    def validate_selected_aeo_prompts(self, value):
        if value is None:
            return []
        from accounts.aeo.prompt_storage import normalize_selected_aeo_prompts_payload

        try:
            out = normalize_selected_aeo_prompts_payload(value)
        except ValueError as exc:
            raise serializers.ValidationError(str(exc)) from exc
        attrs = getattr(self, "initial_data", None) if isinstance(getattr(self, "initial_data", None), dict) else {}
        total_cap = aeo_effective_total_selected_slots_for_validation(getattr(self, "instance", None), attrs or {})
        return out[:total_cap]

    def validate(self, attrs):
        attrs = super().validate(attrs)
        instance = getattr(self, "instance", None)
        effective_reach = str(
            attrs.get(
                "customer_reach",
                getattr(instance, "customer_reach", BusinessProfile.CUSTOMER_REACH_ONLINE),
            )
            or BusinessProfile.CUSTOMER_REACH_ONLINE
        ).strip().lower()
        effective_state = str(
            attrs.get(
                "customer_reach_state",
                getattr(instance, "customer_reach_state", ""),
            )
            or ""
        ).strip()
        if effective_reach == BusinessProfile.CUSTOMER_REACH_LOCAL and not effective_state:
            raise serializers.ValidationError(
                {"customer_reach_state": "State is required when customer_reach is local."}
            )
        if effective_reach != BusinessProfile.CUSTOMER_REACH_LOCAL:
            attrs["customer_reach_state"] = ""
            attrs["customer_reach_city"] = ""
        if "tracked_competitors" in self.initial_data:
            raw = self.initial_data.get("tracked_competitors")
            if raw is None:
                attrs["_tracked_competitors"] = []
            elif not isinstance(raw, list):
                raise serializers.ValidationError(
                    {"tracked_competitors": "Expected a list of objects with name and domain."}
                )
            else:
                attrs["_tracked_competitors"] = self._normalize_tracked_competitors_input(raw)
        if "selected_aeo_prompts" in attrs:
            sp = attrs.get("selected_aeo_prompts")
            if sp is not None:
                from accounts.aeo.prompt_storage import count_custom_prompts_in_selected

                n = len(sp)
                suggested_cap = aeo_effective_cap_for_validation(getattr(self, "instance", None), attrs)
                custom_cap = aeo_effective_custom_prompt_cap_for_validation(getattr(self, "instance", None), attrs)
                custom_n = count_custom_prompts_in_selected(sp)
                suggested_n = n - custom_n
                need_min = aeo_onboarding_min_for_validation(getattr(self, "instance", None), attrs)
                if suggested_n > suggested_cap:
                    raise serializers.ValidationError(
                        {
                            "selected_aeo_prompts": (
                                f"At most {suggested_cap} suggested (non-custom) monitored prompts for your plan; "
                                f"got {suggested_n}."
                            )
                        }
                    )
                if custom_n > custom_cap:
                    raise serializers.ValidationError(
                        {
                            "selected_aeo_prompts": (
                                f"At most {custom_cap} custom monitored prompts for your plan; got {custom_n}."
                            )
                        }
                    )
                if n != 0 and n < need_min:
                    raise serializers.ValidationError(
                        {
                            "selected_aeo_prompts": (
                                f"Must be empty or at least {need_min} prompts (onboarding baseline); got {n}."
                            )
                        }
                    )
        return attrs

    def validate_email(self, value: str) -> str:
        email = (value or "").strip().lower()
        if not email:
            raise serializers.ValidationError("Email is required.")
        user = getattr(getattr(self, "instance", None), "user", None)
        exists = User.objects.filter(email__iexact=email)
        if user is not None:
            exists = exists.exclude(pk=user.pk)
        if exists.exists():
            raise serializers.ValidationError("An account with this email already exists.")
        return email

    @staticmethod
    def _split_full_name(full_name: str) -> tuple[str, str]:
        parts = [p for p in (full_name or "").strip().split() if p]
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], " ".join(parts[1:])

    def create(self, validated_data: dict) -> BusinessProfile:
        tc = validated_data.pop("_tracked_competitors", None)
        instance = super().create(validated_data)
        if tc is not None:
            self._apply_tracked_competitors(instance, tc)
        return instance

    def update(self, instance: BusinessProfile, validated_data: dict) -> BusinessProfile:
        tc = validated_data.pop("_tracked_competitors", None)
        user_data = validated_data.pop("user", {}) or {}
        full_name = validated_data.get("full_name", None)
        user = getattr(instance, "user", None)

        if user is not None:
            email = user_data.get("email", None)
            user_changed_fields: list[str] = []
            if email is not None:
                email = str(email).strip().lower()
                if email and user.email != email:
                    user.email = email
                    user_changed_fields.append("email")
                if email and getattr(user, "username", "") != email:
                    user.username = email
                    user_changed_fields.append("username")
            if full_name is not None:
                first_name, last_name = self._split_full_name(str(full_name))
                if getattr(user, "first_name", "") != first_name:
                    user.first_name = first_name
                    user_changed_fields.append("first_name")
                if getattr(user, "last_name", "") != last_name:
                    user.last_name = last_name
                    user_changed_fields.append("last_name")
            if user_changed_fields:
                user.save(update_fields=sorted(set(user_changed_fields)))

        instance = super().update(instance, validated_data)
        if tc is not None:
            self._apply_tracked_competitors(instance, tc)
        return instance

    def to_representation(self, instance: BusinessProfile) -> dict:
        data = super().to_representation(instance)
        user = getattr(instance, "user", None)
        if user is not None:
            first = str(getattr(user, "first_name", "") or "").strip()
            last = str(getattr(user, "last_name", "") or "").strip()
            full = " ".join([x for x in [first, last] if x]).strip()
            if full:
                data["full_name"] = full
            data["email"] = str(getattr(user, "email", "") or "")
        return data

    def _get_seo_bundle(self, obj: BusinessProfile) -> dict | None:
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return None

        user = getattr(obj, "user", None)
        site_url = obj.website_url or ""

        if not user:
            logger.warning(
                "[BusinessProfileSerializer] get_seo_score: missing user for profile id=%s",
                getattr(obj, "id", None),
            )
            return None

        # Per-profile-id cache: DRF reuses one child serializer for many=True lists; a single
        # attribute would serve every row the first profile's bundle.
        cache_attr = "_seo_bundle_cache_by_profile_id"
        cache_map = getattr(self, cache_attr, None)
        pid = getattr(obj, "id", None)
        if pid is not None and isinstance(cache_map, dict) and pid in cache_map:
            return cache_map[pid]

        try:
            logger.info(
                "[BusinessProfileSerializer] get_seo_score: user_id=%s site_url=%s",
                getattr(user, "id", None),
                site_url,
            )
            data = get_or_refresh_seo_score_for_user(
                user,
                site_url=site_url or None,
                business_profile=obj,
            )
            if not isinstance(cache_map, dict):
                cache_map = {}
            if pid is not None:
                cache_map[pid] = data
                setattr(self, cache_attr, cache_map)
            if not data:
                logger.info(
                    "[BusinessProfileSerializer] get_seo_score: no data returned for user_id=%s",
                    getattr(user, "id", None),
                )
                return None

            logger.info(
                "[BusinessProfileSerializer] get_seo_score: resolved seo_bundle=%s for user_id=%s",
                {k: data.get(k) for k in ["seo_score", "search_performance_score"]},
                getattr(user, "id", None),
            )
            return data
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "[BusinessProfileSerializer] get_seo_score: exception for user_id=%s: %s",
                getattr(user, "id", None),
                str(exc)[:300],
            )
            if pid is not None:
                if not isinstance(cache_map, dict):
                    cache_map = {}
                cache_map[pid] = None
                setattr(self, cache_attr, cache_map)
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

    def get_estimated_search_appearances_monthly(self, obj: BusinessProfile) -> int | None:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return None
        val = bundle.get("estimated_search_appearances_monthly")
        return int(val) if val is not None else None

    def get_top_keywords(self, obj: BusinessProfile):
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            # Cached snapshot only — lets onboarding hydrate keyword topics without DataForSEO.
            snap = (
                SEOOverviewSnapshot.objects.filter(business_profile=obj)
                .order_by("-last_fetched_at", "-id")
                .only("top_keywords")
                .first()
            )
            if snap is not None:
                tk = getattr(snap, "top_keywords", None) or []
                return list(tk) if isinstance(tk, list) else []
            return []
        bundle = self._get_seo_bundle(obj)
        from_bundle: list = []
        if bundle:
            raw = bundle.get("top_keywords")
            if isinstance(raw, list):
                from_bundle = list(raw)
        if from_bundle:
            return from_bundle
        return _fallback_top_keywords_from_stored_snapshots(obj)

    def _get_skip_heavy_seo_actions_payload(self, obj: BusinessProfile) -> dict:
        cache_attr = "_skip_heavy_seo_actions_payload_by_profile_id"
        cache_map = getattr(self, cache_attr, None)
        pid = getattr(obj, "id", None)
        if isinstance(cache_map, dict) and pid in cache_map:
            return cache_map[pid]
        payload = _skip_heavy_latest_seo_snapshot_actions_payload(obj)
        if not isinstance(cache_map, dict):
            cache_map = {}
        cache_map[pid] = payload
        setattr(self, cache_attr, cache_map)
        return payload

    def get_seo_next_steps(self, obj: BusinessProfile):
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return self._get_skip_heavy_seo_actions_payload(obj).get("seo_next_steps") or []
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return []
        return bundle.get("seo_next_steps") or []

    def _get_aeo_bundle(self, obj: BusinessProfile) -> dict:
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return {
                "question_coverage_score": 0,
                "questions_found": [],
                "questions_missing": [],
                "faq_readiness_score": 0,
                "faq_blocks_found": 0,
                "faq_schema_present": False,
                "snippet_readiness_score": 0,
                "answer_blocks_found": 0,
                "aeo_score": 0,
                "aeo_recommendations": [],
                "aeo_status": "skipped",
                "aeo_last_computed_at": None,
            }

        domain = normalize_domain(obj.website_url or "") or ""
        force_aeo_refresh = bool(context.get("force_aeo_refresh"))
        default_location_code = int(getattr(settings, "DATAFORSEO_LOCATION_CODE", 2840))
        resolved_location_code, _, resolved_location_label = get_profile_location_code(obj, default_location_code)
        cache_key = f"{getattr(obj, 'id', 'unknown')}:{domain}:{int(resolved_location_code)}"
        cache_attr = "_aeo_bundle_cache_map"
        cache_map = getattr(self, cache_attr, {})
        if (not force_aeo_refresh) and (cache_key in cache_map):
            # #region agent log
            _debug.log(
                "serializers.py:BusinessProfileSerializer:_get_aeo_bundle:cache_hit",
                "AEO bundle cache hit in serializer",
                {"profile_id": getattr(obj, "id", None), "cache_key": cache_key},
                "H1",
            )
            # #endregion
            return cache_map[cache_key]

        domain = domain or None
        niche = (obj.industry or "").strip() or None
        # #region agent log
        _debug.log(
            "serializers.py:BusinessProfileSerializer:_get_aeo_bundle:entry",
            "Resolving AEO bundle for profile",
            {
                "profile_id": getattr(obj, "id", None),
                "user_id": getattr(getattr(obj, "user", None), "id", None),
                "has_domain": bool(domain),
                "industry": niche or "",
                "force_aeo_refresh": force_aeo_refresh,
            },
            "H1",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H1 serializer entry profile_id=%s user_id=%s domain=%s force_refresh=%s industry=%s",
            getattr(obj, "id", None),
            getattr(getattr(obj, "user", None), "id", None),
            domain or "",
            force_aeo_refresh,
            niche or "",
        )

        if not force_aeo_refresh:
            snapshot = AEOOverviewSnapshot.objects.filter(
                profile=obj,
                domain=domain or "",
                location_code=int(resolved_location_code),
            ).first()
            if snapshot and (timezone.now() - snapshot.refreshed_at) <= AEO_SNAPSHOT_TTL:
                # #region agent log
                _debug.log(
                    "serializers.py:BusinessProfileSerializer:_get_aeo_bundle:snapshot_cache_hit",
                    "Returning AEO snapshot cache",
                    {
                        "profile_id": getattr(obj, "id", None),
                        "domain": domain or "",
                        "snapshot_refreshed_at": str(snapshot.refreshed_at),
                        "snapshot_aeo_score": int(snapshot.aeo_score or 0),
                        "snapshot_question_coverage_score": int(snapshot.question_coverage_score or 0),
                        "snapshot_faq_readiness_score": int(snapshot.faq_readiness_score or 0),
                        "snapshot_snippet_readiness_score": int(snapshot.snippet_readiness_score or 0),
                        "ttl_seconds": int(AEO_SNAPSHOT_TTL.total_seconds()),
                    },
                    "H1",
                )
                # #endregion
                logger.info(
                    "[AEO debug 442421] snapshot cache hit profile_id=%s domain=%s refreshed_at=%s aeo=%s q=%s faq=%s snip=%s found=%s missing=%s",
                    getattr(obj, "id", None),
                    domain or "",
                    snapshot.refreshed_at,
                    int(snapshot.aeo_score or 0),
                    int(snapshot.question_coverage_score or 0),
                    int(snapshot.faq_readiness_score or 0),
                    int(snapshot.snippet_readiness_score or 0),
                    len(snapshot.questions_found or []),
                    len(snapshot.questions_missing or []),
                )
                safe_bundle = {
                    "question_coverage_score": int(snapshot.question_coverage_score or 0),
                    "questions_found": list(snapshot.questions_found or []),
                    "questions_missing": list(snapshot.questions_missing or []),
                    "faq_readiness_score": int(snapshot.faq_readiness_score or 0),
                    "faq_blocks_found": int(snapshot.faq_blocks_found or 0),
                    "faq_schema_present": bool(snapshot.faq_schema_present),
                    "snippet_readiness_score": int(snapshot.snippet_readiness_score or 0),
                    "answer_blocks_found": int(snapshot.answer_blocks_found or 0),
                    "aeo_score": int(snapshot.aeo_score or 0),
                    "aeo_recommendations": list(snapshot.aeo_recommendations or [])[:5],
                    "aeo_status": "ready",
                    "aeo_last_computed_at": snapshot.refreshed_at.isoformat() if snapshot.refreshed_at else None,
                }
                cache_map[cache_key] = safe_bundle
                setattr(self, cache_attr, cache_map)
                return safe_bundle

        try:
            from accounts.third_party_usage import usage_profile_context

            with usage_profile_context(obj):
                data = get_aeo_content_readiness_for_site(
                    target_domain=domain,
                    niche=niche,
                    cache_key_seed=f"user:{getattr(getattr(obj, 'user', None), 'id', 'unknown')}",
                    force_refresh=force_aeo_refresh,
                    location_code=resolved_location_code,
                    profile_id=getattr(obj, "id", None),
                ) or {}
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "[BusinessProfileSerializer] get_aeo_bundle failed for profile id=%s",
                getattr(obj, "id", None),
            )
            data = {}
        # #region agent log
        _debug.log(
            "serializers.py:BusinessProfileSerializer:_get_aeo_bundle:result",
            "AEO bundle resolved from helper",
            {
                "profile_id": getattr(obj, "id", None),
                "question_coverage_score": int(data.get("question_coverage_score") or 0),
                "faq_readiness_score": int(data.get("faq_readiness_score") or 0),
                "snippet_readiness_score": int(data.get("snippet_readiness_score") or 0),
                "questions_found_count": len(data.get("questions_found") or []),
                "questions_missing_count": len(data.get("questions_missing") or []),
                "faq_blocks_found": int(data.get("faq_blocks_found") or 0),
                "answer_blocks_found": int(data.get("answer_blocks_found") or 0),
            },
            "H2",
        )
        # #endregion
        logger.info(
            "[AEO debug eb0539] H2 serializer result profile_id=%s q=%s faq=%s snip=%s found=%s missing=%s",
            getattr(obj, "id", None),
            int(data.get("question_coverage_score") or 0),
            int(data.get("faq_readiness_score") or 0),
            int(data.get("snippet_readiness_score") or 0),
            len(data.get("questions_found") or []),
            len(data.get("questions_missing") or []),
        )

        safe_bundle = {
            "question_coverage_score": int(data.get("question_coverage_score") or 0),
            "questions_found": data.get("questions_found") or [],
            "questions_missing": data.get("questions_missing") or [],
            "faq_readiness_score": int(data.get("faq_readiness_score") or 0),
            "faq_blocks_found": int(data.get("faq_blocks_found") or 0),
            "faq_schema_present": bool(data.get("faq_schema_present") or False),
            "snippet_readiness_score": int(data.get("snippet_readiness_score") or 0),
            "answer_blocks_found": int(data.get("answer_blocks_found") or 0),
            "aeo_status": str(data.get("aeo_status") or "ready"),
            "aeo_last_computed_at": data.get("aeo_last_computed_at"),
        }
        aeo_status = str(safe_bundle.get("aeo_status") or "ready")
        if aeo_status != "ready":
            safe_bundle["aeo_score"] = int(data.get("aeo_score") or 0)
            safe_bundle["aeo_recommendations"] = list(data.get("aeo_recommendations") or [])
            cache_map[cache_key] = safe_bundle
            setattr(self, cache_attr, cache_map)
            return safe_bundle

        question_coverage_score = int(safe_bundle.get("question_coverage_score") or 0)
        faq_readiness_score = int(safe_bundle.get("faq_readiness_score") or 0)
        snippet_readiness_score = int(safe_bundle.get("snippet_readiness_score") or 0)
        # Use pipeline-provided AEO score when available (crawl-derived),
        # and only fall back to local weighted composition if missing.
        pipeline_score = data.get("aeo_score")
        if pipeline_score is not None:
            try:
                aeo_score = int(round(float(pipeline_score)))
            except (TypeError, ValueError):
                aeo_score = int(
                    round(
                        (question_coverage_score * 0.4)
                        + (faq_readiness_score * 0.3)
                        + (snippet_readiness_score * 0.3),
                    )
                )
        else:
            aeo_score = int(
                round(
                    (question_coverage_score * 0.4)
                    + (faq_readiness_score * 0.3)
                    + (snippet_readiness_score * 0.3),
                )
            )
        aeo_score = max(0, min(100, aeo_score))

        recommendations: list[str] = []
        existing_snapshot = AEOOverviewSnapshot.objects.filter(
            profile=obj,
            domain=domain or "",
            location_code=int(resolved_location_code),
        ).first()
        should_refresh_recommendations = True
        if (
            existing_snapshot
            and existing_snapshot.aeo_recommendations
            and existing_snapshot.aeo_recommendations_refreshed_at
            and (timezone.now() - existing_snapshot.aeo_recommendations_refreshed_at) <= AEO_RECOMMENDATIONS_TTL
        ):
            should_refresh_recommendations = False
            recommendations = list(existing_snapshot.aeo_recommendations or [])[:5]
        if should_refresh_recommendations:
            context = getattr(self, "context", {}) or {}
            if bool(context.get("disable_seo_context_for_aeo")):
                logger.warning(
                    "[AEO guard] disable_seo_context_for_aeo enabled; skipping SEO helper for profile id=%s",
                    getattr(obj, "id", None),
                )
                seo_bundle = {}
            else:
                seo_bundle = self._get_seo_bundle(obj) or {}
            recommendations = generate_aeo_recommendations(safe_bundle, seo_bundle)[:5]
        if not recommendations:
            # Minimal safe fallback if OpenAI fails.
            recommendations = [
                "Publish a dedicated FAQ page targeting your highest-volume missing questions.",
                "Add concise 40-60 word answer blocks directly below question-style headings.",
                "Implement or validate FAQPage schema on core service pages.",
                "Expand content on pages with high-volume keywords where rank is weak or missing.",
                "Refresh internal linking so top-demand question pages are linked from navigation.",
            ]
        recommendations_refreshed_at = timezone.now() if should_refresh_recommendations else (
            existing_snapshot.aeo_recommendations_refreshed_at if existing_snapshot else timezone.now()
        )

        safe_bundle["aeo_score"] = aeo_score
        safe_bundle["aeo_recommendations"] = recommendations[:5]
        # #region agent log
        _debug.log(
            "serializers.py:BusinessProfileSerializer:_get_aeo_bundle:computed_bundle",
            "Computed AEO bundle before snapshot save",
            {
                "profile_id": getattr(obj, "id", None),
                "aeo_score": int(aeo_score),
                "question_coverage_score": int(question_coverage_score),
                "faq_readiness_score": int(faq_readiness_score),
                "snippet_readiness_score": int(snippet_readiness_score),
                "questions_found_count": len(safe_bundle.get("questions_found") or []),
                "questions_missing_count": len(safe_bundle.get("questions_missing") or []),
            },
            "H2",
        )
        # #endregion

        AEOOverviewSnapshot.objects.update_or_create(
            profile=obj,
            domain=domain or "",
            location_code=int(resolved_location_code),
            defaults={
                "user": obj.user,
                "location_label": resolved_location_label or "",
                "niche": niche or "",
                "aeo_score": int(safe_bundle["aeo_score"] or 0),
                "question_coverage_score": int(safe_bundle["question_coverage_score"] or 0),
                "questions_found": list(safe_bundle["questions_found"] or []),
                "questions_missing": list(safe_bundle["questions_missing"] or []),
                "faq_readiness_score": int(safe_bundle["faq_readiness_score"] or 0),
                "faq_blocks_found": int(safe_bundle["faq_blocks_found"] or 0),
                "faq_schema_present": bool(safe_bundle["faq_schema_present"]),
                "snippet_readiness_score": int(safe_bundle["snippet_readiness_score"] or 0),
                "answer_blocks_found": int(safe_bundle["answer_blocks_found"] or 0),
                "aeo_recommendations": list(safe_bundle["aeo_recommendations"] or []),
                "aeo_recommendations_refreshed_at": recommendations_refreshed_at,
            },
        )
        cache_map[cache_key] = safe_bundle
        setattr(self, cache_attr, cache_map)
        return safe_bundle

    def get_aeo_score(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("aeo_score") or 0)

    def get_question_coverage_score(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("question_coverage_score") or 0)

    def get_questions_found(self, obj: BusinessProfile):
        bundle = self._get_aeo_bundle(obj)
        return bundle.get("questions_found") or []

    def get_questions_missing(self, obj: BusinessProfile):
        bundle = self._get_aeo_bundle(obj)
        return bundle.get("questions_missing") or []

    def get_faq_readiness_score(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("faq_readiness_score") or 0)

    def get_faq_blocks_found(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("faq_blocks_found") or 0)

    def get_faq_schema_present(self, obj: BusinessProfile) -> bool:
        bundle = self._get_aeo_bundle(obj)
        return bool(bundle.get("faq_schema_present") or False)

    def get_snippet_readiness_score(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("snippet_readiness_score") or 0)

    def get_answer_blocks_found(self, obj: BusinessProfile) -> int:
        bundle = self._get_aeo_bundle(obj)
        return int(bundle.get("answer_blocks_found") or 0)

    def get_aeo_recommendations(self, obj: BusinessProfile):
        bundle = self._get_aeo_bundle(obj)
        return bundle.get("aeo_recommendations") or []

    def get_aeo_status(self, obj: BusinessProfile) -> str:
        bundle = self._get_aeo_bundle(obj)
        return str(bundle.get("aeo_status") or "ready")

    def get_aeo_last_computed_at(self, obj: BusinessProfile):
        bundle = self._get_aeo_bundle(obj)
        return bundle.get("aeo_last_computed_at")

    def get_keyword_action_suggestions(self, obj: BusinessProfile):
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return self._get_skip_heavy_seo_actions_payload(obj).get("keyword_action_suggestions") or []
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return []
        # List of {\"keyword\": str, \"suggestion\": str}
        return bundle.get("keyword_action_suggestions") or []

    def get_seo_structured_issues(self, obj: BusinessProfile):
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return self._get_skip_heavy_seo_actions_payload(obj).get("seo_structured_issues") or []
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return []
        raw = bundle.get("seo_structured_issues")
        return list(raw) if isinstance(raw, list) else []

    def get_enrichment_status(self, obj: BusinessProfile) -> str:
        context = getattr(self, "context", {}) or {}
        if context.get("skip_heavy_profile_metrics"):
            return str(self._get_skip_heavy_seo_actions_payload(obj).get("enrichment_status") or "complete")
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return "complete"
        return bundle.get("enrichment_status") or "complete"

    def get_seo_metrics_location_mode(self, obj: BusinessProfile) -> str:
        bundle = self._get_seo_bundle(obj)
        if bundle and bundle.get("seo_metrics_location_mode") is not None:
            return str(bundle.get("seo_metrics_location_mode") or "organic")
        return str(getattr(obj, "seo_location_mode", None) or "organic")

    def get_seo_location_label(self, obj: BusinessProfile) -> str:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return ""
        return str(bundle.get("seo_location_label") or "")

    def get_local_verified_keyword_count(self, obj: BusinessProfile) -> int:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return 0
        return int(bundle.get("local_verified_keyword_count") or 0)

    def get_local_verification_affects_visibility(self, obj: BusinessProfile) -> bool:
        bundle = self._get_seo_bundle(obj)
        if not bundle:
            return False
        return bool(bundle.get("local_verification_affects_visibility"))


class BusinessProfileSEOSerializer(BusinessProfileSerializer):
    class Meta(BusinessProfileSerializer.Meta):
        fields = [
            "id",
            "email",
            "full_name",
            "business_name",
            "business_address",
            "industry",
            "phone",
            "description",
            "website_url",
            "plan",
            "is_main",
            "tracked_competitors",
            "seo_competitor_domains_override",
            "seo_location_mode",
            "seo_score",
            "search_performance_score",
            "search_visibility_percent",
            "missed_searches_monthly",
            "total_search_volume",
            "estimated_search_appearances_monthly",
            "organic_visitors",
            "top_keywords",
            "seo_next_steps",
            "keyword_action_suggestions",
            "seo_structured_issues",
            "enrichment_status",
            "seo_metrics_location_mode",
            "seo_location_label",
            "local_verified_keyword_count",
            "local_verification_affects_visibility",
            "created_at",
            "updated_at",
            "viewer_team_role",
            "viewer_is_main_account_owner",
            "viewer_can_edit_company_profile",
            "viewer_can_access_billing",
            "viewer_can_manage_team",
            "viewer_email",
        ]

    def _get_aeo_bundle(self, obj: BusinessProfile) -> dict:
        logger.warning(
            "[SEO guard] AEO bundle requested from SEO serializer for profile id=%s; returning empty",
            getattr(obj, "id", None),
        )
        return {}


class BusinessProfileAEOSerializer(BusinessProfileSerializer):
    class Meta(BusinessProfileSerializer.Meta):
        fields = [
            "id",
            "email",
            "full_name",
            "business_name",
            "business_address",
            "industry",
            "phone",
            "description",
            "website_url",
            "selected_aeo_prompts",
            "aeo_onboarding_prompt_target_count",
            "aeo_prompt_expansion_status",
            "aeo_prompt_expansion_target",
            "aeo_prompt_expansion_progress",
            "aeo_prompt_expansion_last_error",
            "aeo_prompt_expansion_updated_at",
            "plan",
            "is_main",
            "tracked_competitors",
            "seo_location_mode",
            "aeo_score",
            "question_coverage_score",
            "questions_found",
            "questions_missing",
            "faq_readiness_score",
            "faq_blocks_found",
            "faq_schema_present",
            "snippet_readiness_score",
            "answer_blocks_found",
            "aeo_recommendations",
            "aeo_status",
            "aeo_last_computed_at",
            "created_at",
            "updated_at",
            "viewer_team_role",
            "viewer_is_main_account_owner",
            "viewer_can_edit_company_profile",
            "viewer_can_access_billing",
            "viewer_can_manage_team",
            "viewer_email",
        ]

    def _get_seo_bundle(self, obj: BusinessProfile) -> dict | None:
        logger.warning(
            "[AEO guard] SEO bundle requested from AEO serializer for profile id=%s; returning none",
            getattr(obj, "id", None),
        )
        return None
