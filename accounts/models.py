from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from .domain_utils import normalize_tracked_competitor_domain


class TrackedCompetitor(models.Model):
    """
    Canonical competitor identity by normalized base domain.

    Multiple business profiles may reference the same row; this model does not
    record which profiles use it (no reverse relation from competitor → profiles).
    """

    name = models.CharField(max_length=255)
    domain = models.CharField(
        max_length=253,
        unique=True,
        db_index=True,
        help_text="Normalized base host (no path, no scheme, lowercase, no leading www.).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Tracked competitor"
        verbose_name_plural = "Tracked competitors"
        ordering = ("domain",)

    def __str__(self) -> str:
        return f"{self.name} ({self.domain})"

    def clean(self) -> None:
        nd = normalize_tracked_competitor_domain(self.domain)
        if not nd:
            raise ValidationError({"domain": "Enter a valid domain."})
        self.domain = nd
        if not (self.name or "").strip():
            raise ValidationError({"name": "Name is required."})

    def save(self, *args, **kwargs) -> None:
        self.full_clean()
        super().save(*args, **kwargs)


class BusinessProfile(models.Model):
    CUSTOMER_REACH_ONLINE = "online"
    CUSTOMER_REACH_LOCAL = "local"
    CUSTOMER_REACH_CHOICES = [
        (CUSTOMER_REACH_ONLINE, "Online"),
        (CUSTOMER_REACH_LOCAL, "Local"),
    ]

    SEO_LOCATION_MODE_ORGANIC = "organic"
    SEO_LOCATION_MODE_LOCAL = "local"
    SEO_LOCATION_MODE_CHOICES = [
        (SEO_LOCATION_MODE_ORGANIC, "Organic"),
        (SEO_LOCATION_MODE_LOCAL, "Local"),
    ]

    """
    Stores business profile settings for a user.

    Each user can have multiple BusinessProfiles. One of them can be marked
    as the main profile (used by default in most places).
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="business_profiles",
    )

    full_name = models.CharField(max_length=255, blank=True)
    business_name = models.CharField(max_length=255, blank=True)
    business_address = models.CharField(max_length=255, blank=True)
    customer_reach = models.CharField(
        max_length=16,
        choices=CUSTOMER_REACH_CHOICES,
        default=CUSTOMER_REACH_ONLINE,
    )
    customer_reach_state = models.CharField(max_length=64, blank=True, default="")
    customer_reach_city = models.CharField(max_length=128, blank=True, default="")

    # Industry is intentionally a free-text field (no choices).
    industry = models.CharField(max_length=255, blank=True)

    PLAN_NONE = ""
    PLAN_STARTER = "starter"
    PLAN_PRO = "pro"
    PLAN_ADVANCED = "advanced"
    PLAN_CHOICES = [
        (PLAN_NONE, "No paid plan"),
        (PLAN_STARTER, "Starter"),
        (PLAN_PRO, "Pro"),
        (PLAN_ADVANCED, "Advanced"),
    ]
    plan = models.CharField(
        max_length=16,
        choices=PLAN_CHOICES,
        default=PLAN_NONE,
        blank=True,
    )
    stripe_customer_id = models.CharField(max_length=255, blank=True, default="")
    stripe_subscription_id = models.CharField(max_length=255, blank=True, default="")
    stripe_price_id = models.CharField(max_length=255, blank=True, default="")
    stripe_subscription_status = models.CharField(max_length=64, blank=True, default="")
    stripe_current_period_end = models.DateTimeField(null=True, blank=True)
    stripe_cancel_at_period_end = models.BooleanField(default=False)

    phone = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    website_url = models.URLField(blank=True)

    # Whether this is the main / primary profile for the user.
    is_main = models.BooleanField(default=False)

    # Optional: comma-separated competitor domains for SEO keyword gap analysis.
    # When provided, these are preferred over DataForSEO auto-competitors.
    # Example: "smilebright.com, nocavityclinic.com, oakdentalcare.com"
    seo_competitor_domains_override = models.TextField(blank=True, default="")
    tracked_competitors = models.ManyToManyField(
        TrackedCompetitor,
        blank=True,
        related_name="+",
    )
    seo_location_mode = models.CharField(
        max_length=16,
        choices=SEO_LOCATION_MODE_CHOICES,
        default=SEO_LOCATION_MODE_ORGANIC,
    )
    # Persisted SEO location context (used by DataForSEO helpers via getattr on profile).
    seo_location_depth = models.IntegerField(
        default=0,
        help_text="Legacy; prefer seo_location_mode. Kept for DB compatibility and older clients.",
    )
    seo_location_code = models.IntegerField(
        default=0,
        help_text="DataForSEO location code when seo_location_mode is local; 0 means unset / use default.",
    )
    seo_location_label = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Human-readable location label for local SEO mode.",
    )

    # Onboarding / settings: ordered list of AEO visibility prompt strings the user chose to track.
    selected_aeo_prompts = models.JSONField(default=list, blank=True)

    AEO_PROMPT_EXPANSION_IDLE = "idle"
    AEO_PROMPT_EXPANSION_QUEUED = "queued"
    AEO_PROMPT_EXPANSION_RUNNING = "running"
    AEO_PROMPT_EXPANSION_COMPLETE = "complete"
    AEO_PROMPT_EXPANSION_ERROR = "error"
    AEO_PROMPT_EXPANSION_STATUS_CHOICES = [
        (AEO_PROMPT_EXPANSION_IDLE, "Idle"),
        (AEO_PROMPT_EXPANSION_QUEUED, "Queued"),
        (AEO_PROMPT_EXPANSION_RUNNING, "Running"),
        (AEO_PROMPT_EXPANSION_COMPLETE, "Complete"),
        (AEO_PROMPT_EXPANSION_ERROR, "Error"),
    ]
    aeo_prompt_expansion_status = models.CharField(
        max_length=16,
        choices=AEO_PROMPT_EXPANSION_STATUS_CHOICES,
        default=AEO_PROMPT_EXPANSION_IDLE,
    )
    aeo_prompt_expansion_target = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Monitored prompt cap when expansion last ran (plan-derived).",
    )
    aeo_prompt_expansion_progress = models.PositiveIntegerField(
        default=0,
        help_text="len(selected_aeo_prompts) snapshot after last expansion step.",
    )
    aeo_prompt_expansion_last_error = models.TextField(blank=True, default="")
    aeo_prompt_expansion_updated_at = models.DateTimeField(null=True, blank=True)
    aeo_full_phase_eta_state = models.JSONField(
        default=dict,
        blank=True,
        help_text="Rolling completion durations + recorded prompt hashes for AI Visibility ETA.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Business profile"
        verbose_name_plural = "Business profiles"

    def __str__(self) -> str:
        return f"BusinessProfile(user={self.user!s})"


class BusinessProfileMembership(models.Model):
    """
    Links a User to a BusinessProfile (team workspace).

    The main billing account holder is marked ``is_owner=True`` (one per profile).
    ``role`` distinguishes admins (full product access) from read-only members.
    """

    ROLE_ADMIN = "admin"
    ROLE_MEMBER = "member"
    ROLE_CHOICES = [
        (ROLE_ADMIN, "Admin"),
        (ROLE_MEMBER, "Member"),
    ]

    business_profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="team_memberships",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="business_profile_memberships",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default=ROLE_MEMBER)
    is_owner = models.BooleanField(
        default=False,
        help_text="True for the primary account holder (billing owner) of this business profile.",
    )
    hidden_from_team_ui = models.BooleanField(
        default=False,
        help_text=(
            "When true, keep this membership for access control but hide it from "
            "customer-facing team member lists."
        ),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("business_profile", "user")
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["business_profile"]),
        ]

    def __str__(self) -> str:
        return f"BusinessProfileMembership(profile={self.business_profile_id}, user={self.user_id})"


class SEOOverviewSnapshot(models.Model):
    """
    Stores monthly SEO overview metrics per business profile
    (first day of the month + last time data was fetched).
    Keyword list and search metrics are cached and only refreshed once per hour
    or when the profile's website URL (domain) changes.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="seo_overview_snapshots",
    )
    business_profile = models.ForeignKey(
        "BusinessProfile",
        on_delete=models.CASCADE,
        related_name="seo_overview_snapshots",
    )
    period_start = models.DateField()
    last_fetched_at = models.DateTimeField(auto_now=True)

    organic_visitors = models.IntegerField(default=0)
    prev_organic_visitors = models.IntegerField(default=0)
    keywords_ranking = models.IntegerField(default=0)
    top3_positions = models.IntegerField(default=0)

    # Cache for keyword list and search metrics; refreshed at most once per hour or when URL changes
    refreshed_at = models.DateTimeField(null=True, blank=True)
    cached_domain = models.CharField(max_length=255, blank=True)
    cached_location_mode = models.CharField(max_length=16, blank=True, default="organic")
    cached_location_code = models.IntegerField(default=0)
    cached_location_label = models.CharField(max_length=255, blank=True, default="")
    local_verification_applied = models.BooleanField(default=False)
    local_verified_keyword_count = models.IntegerField(default=0)
    top_keywords = models.JSONField(default=list, blank=True)
    total_search_volume = models.IntegerField(default=0)
    estimated_search_appearances_monthly = models.IntegerField(default=0)
    missed_searches_monthly = models.IntegerField(default=0)
    search_visibility_percent = models.IntegerField(default=0)
    search_performance_score = models.IntegerField(null=True, blank=True)
    seo_next_steps = models.JSONField(default=list, blank=True)
    seo_next_steps_refreshed_at = models.DateTimeField(null=True, blank=True)
    # When keyword enrichment (gap + LLM) last completed; used for duplicate prevention and enrichment_status.
    keywords_enriched_at = models.DateTimeField(null=True, blank=True)
    # Per-keyword action suggestions generated by OpenAI, for \"Do these now\" UI.
    keyword_action_suggestions = models.JSONField(default=list, blank=True)
    keyword_action_suggestions_refreshed_at = models.DateTimeField(null=True, blank=True)
    # Deterministic SEO issues (accounts.seo.seo_issue_engine.build_structured_issues) for audit / admin.
    seo_structured_issues = models.JSONField(default=list, blank=True)
    seo_structured_issues_refreshed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = (
            "business_profile",
            "period_start",
            "cached_location_mode",
            "cached_location_code",
        )
        verbose_name = "SEO overview snapshot"
        verbose_name_plural = "SEO overview snapshots"

    def __str__(self) -> str:
        return (
            f"SEOOverviewSnapshot(profile_id={getattr(self, 'business_profile_id', None)}, "
            f"period_start={self.period_start})"
        )

    def save(self, *args, **kwargs):
        """
        Enforce canonical SEO metric invariants at write time.
        """
        total = max(0, int(self.total_search_volume or 0))
        appearances = max(0, int(self.estimated_search_appearances_monthly or 0))
        if appearances > total:
            appearances = total
            self.estimated_search_appearances_monthly = appearances

        if total > 0:
            self.search_visibility_percent = int(round((appearances / total) * 100))
        else:
            self.search_visibility_percent = 0

        self.search_visibility_percent = max(0, min(100, int(self.search_visibility_percent or 0)))
        self.missed_searches_monthly = max(0, total - appearances)

        super().save(*args, **kwargs)


class AEOOverviewSnapshot(models.Model):
    """
    Cached AEO readiness metrics per profile/domain/location.
    Used for serializer-level reuse and basic historical/audit visibility.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="aeo_overview_snapshots",
    )
    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_overview_snapshots",
    )
    domain = models.CharField(max_length=255, blank=True)
    location_code = models.IntegerField(default=2840)
    location_label = models.CharField(max_length=255, blank=True, default="")
    niche = models.CharField(max_length=255, blank=True, default="")

    aeo_score = models.IntegerField(default=0)
    question_coverage_score = models.IntegerField(default=0)
    questions_found = models.JSONField(default=list, blank=True)
    questions_missing = models.JSONField(default=list, blank=True)
    faq_readiness_score = models.IntegerField(default=0)
    faq_blocks_found = models.IntegerField(default=0)
    faq_schema_present = models.BooleanField(default=False)
    snippet_readiness_score = models.IntegerField(default=0)
    answer_blocks_found = models.IntegerField(default=0)
    aeo_recommendations = models.JSONField(default=list, blank=True)
    aeo_recommendations_refreshed_at = models.DateTimeField(null=True, blank=True)

    refreshed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("profile", "domain", "location_code")
        verbose_name = "AEO overview snapshot"
        verbose_name_plural = "AEO overview snapshots"

    def __str__(self) -> str:
        return (
            f"AEOOverviewSnapshot(profile_id={self.profile_id}, "
            f"domain={self.domain}, location_code={self.location_code})"
        )


class AEOCompetitorSnapshot(models.Model):
    """
    Cached competitor visibility distribution for a profile and scope.

    ``rows_json`` shape:
      [{domain, display_name, appearances, visibility_pct, rank, last_seen_at}, ...]
    """

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_competitor_snapshots",
    )
    platform_scope = models.CharField(
        max_length=32,
        default="all",
        help_text="all | openai | gemini | perplexity (matches AEOResponseSnapshot.platform).",
    )
    window_start = models.DateTimeField(null=True, blank=True)
    window_end = models.DateTimeField(null=True, blank=True)
    total_slots = models.IntegerField(default=0)
    rows_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "AEO competitor snapshot"
        verbose_name_plural = "AEO competitor snapshots"
        constraints = [
            models.UniqueConstraint(
                fields=["profile", "platform_scope", "window_start", "window_end"],
                name="accounts_aeo_comp_snapshot_scope_uq",
            ),
        ]
        indexes = [
            models.Index(
                fields=["profile", "-updated_at"],
                name="aeo_comp_prof_upd_idx",
            ),
            models.Index(
                fields=["profile", "platform_scope"],
                name="aeo_comp_prof_plat_idx",
            ),
        ]

    def __str__(self) -> str:
        return (
            f"AEOCompetitorSnapshot(profile_id={self.profile_id}, platform={self.platform_scope}, "
            f"slots={self.total_slots})"
        )


class AEODashboardBundleCache(models.Model):
    """
    Cached JSON from ``_build_aeo_prompt_coverage_payload`` for fast dashboard + platform visibility.

    Refreshed synchronously on cache miss and asynchronously when stale (see tasks / views).
    """

    profile = models.OneToOneField(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_dashboard_bundle_cache",
    )
    payload_json = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "AEO dashboard bundle cache"
        verbose_name_plural = "AEO dashboard bundle caches"

    def __str__(self) -> str:
        return f"AEODashboardBundleCache(profile_id={self.profile_id})"


class ActionsGeneratedPageSnapshot(models.Model):
    """
    Persisted structured JSON for Actions → “Generate Page” previews (OpenAI output, no HTML).

    One row per business profile + action_key (client card id). Re-open returns ``page_data``
    when ``content_hash`` matches the request and ``regenerate`` is false. Run ``makemigrations``
    / ``migrate`` to create or alter the table.
    """

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="generated_page_snapshots",
    )
    action_key = models.CharField(
        max_length=512,
        db_index=True,
        help_text="Actions UI card id (e.g. seo-step-0, aeo strategy stable id).",
    )
    content_hash = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True,
        help_text="SHA-256 hex of canonical generation inputs; cache hit only when this matches the request.",
    )
    page_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Actions generated page snapshot"
        verbose_name_plural = "Actions generated page snapshots"
        constraints = [
            models.UniqueConstraint(
                fields=["profile", "action_key"],
                name="uniq_actions_genpage_profile_action",
            ),
        ]

    def __str__(self) -> str:
        return f"ActionsGeneratedPageSnapshot(profile_id={self.profile_id}, key={self.action_key[:40]!r})"


class AEOResponseSnapshot(models.Model):
    """
    Raw OpenAI (or other platform) answer for a single AEO visibility prompt.

    One row per prompt × provider execution; prompt_hash enables reruns and trend comparison.
    Rows from the same dual-provider call share execution_pair_id and execution_run.
    """

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_response_snapshots",
    )
    execution_run = models.ForeignKey(
        "AEOExecutionRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="response_snapshots",
        help_text="Pipeline run that produced this row (if any).",
    )
    execution_pair_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Shared by OpenAI + Gemini snapshots from the same logical prompt execution.",
    )
    prompt_text = models.TextField()
    prompt_type = models.CharField(max_length=32, blank=True, default="")
    is_custom_prompt = models.BooleanField(
        default=False,
        help_text="True when this snapshot was produced from a user-added custom monitored prompt.",
    )
    weight = models.FloatField(default=1.0)
    is_dynamic = models.BooleanField(default=False)
    platform = models.CharField(max_length=64, default="openai")
    model_name = models.CharField(max_length=128, blank=True, default="")
    raw_response = models.TextField(blank=True, default="")
    prompt_hash = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO response snapshot"
        verbose_name_plural = "AEO response snapshots"
        indexes = [
            models.Index(
                fields=["profile", "prompt_hash", "created_at"],
                name="accounts_aeo_rsp_ph_cr",
            ),
        ]
        constraints = []

    def __str__(self) -> str:
        short = (self.prompt_hash or "")[:12]
        return f"AEOResponseSnapshot(profile_id={self.profile_id}, hash={short}...)"


class AEOExtractionSnapshot(models.Model):
    """
    Structured fields extracted from a raw AEOResponseSnapshot via second-pass LLM.

    Always tied to one response row for debugging and scoring pipelines.
    """

    MENTION_TOP = "top"
    MENTION_MIDDLE = "middle"
    MENTION_BOTTOM = "bottom"
    MENTION_NONE = "none"
    MENTION_POSITION_CHOICES = [
        (MENTION_TOP, "Top"),
        (MENTION_MIDDLE, "Middle"),
        (MENTION_BOTTOM, "Bottom"),
        (MENTION_NONE, "None"),
    ]

    SENTIMENT_POSITIVE = "positive"
    SENTIMENT_NEUTRAL = "neutral"
    SENTIMENT_NEGATIVE = "negative"
    SENTIMENT_CHOICES = [
        (SENTIMENT_POSITIVE, "Positive"),
        (SENTIMENT_NEUTRAL, "Neutral"),
        (SENTIMENT_NEGATIVE, "Negative"),
    ]

    response_snapshot = models.ForeignKey(
        AEOResponseSnapshot,
        on_delete=models.CASCADE,
        related_name="extraction_snapshots",
    )
    brand_mentioned = models.BooleanField(default=False)
    mention_position = models.CharField(
        max_length=16,
        choices=MENTION_POSITION_CHOICES,
        default=MENTION_NONE,
    )
    mention_count = models.IntegerField(default=0)
    competitors_json = models.JSONField(default=list, blank=True)
    citations_json = models.JSONField(default=list, blank=True)
    sentiment = models.CharField(
        max_length=16,
        choices=SENTIMENT_CHOICES,
        default=SENTIMENT_NEUTRAL,
    )
    confidence_score = models.FloatField(null=True, blank=True)
    extraction_model = models.CharField(max_length=128, blank=True, default="")
    extraction_parse_failed = models.BooleanField(
        default=False,
        help_text="True when JSON could not be parsed after retry; row stores safe defaults.",
    )

    URL_STATUS_MATCHED = "matched"
    URL_STATUS_NOT_MENTIONED = "not_mentioned"
    URL_STATUS_MENTIONED_URL_WRONG_LIVE = "mentioned_url_wrong_live"
    URL_STATUS_MENTIONED_URL_WRONG_BROKEN = "mentioned_url_wrong_broken"
    BRAND_MENTIONED_URL_STATUS_CHOICES = [
        (URL_STATUS_MATCHED, "Matched canonical domain"),
        (URL_STATUS_NOT_MENTIONED, "Not mentioned / no wrong-URL signal"),
        (URL_STATUS_MENTIONED_URL_WRONG_LIVE, "Mentioned; wrong URL appears live"),
        (URL_STATUS_MENTIONED_URL_WRONG_BROKEN, "Mentioned; wrong URL broken / non-resolving"),
    ]
    brand_mentioned_url_status = models.CharField(
        max_length=40,
        choices=BRAND_MENTIONED_URL_STATUS_CHOICES,
        null=True,
        blank=True,
        db_index=True,
        help_text=(
            "How the model's attributed URL relates to BusinessProfile.website_url. "
            "Null on legacy rows or failed parses. Phase 4 visibility still uses brand_mentioned + "
            "competitor URL match only (wrong-URL cases stay uncredited numerically)."
        ),
    )
    cited_domain_or_url = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="Normalized domain or URL the model tied to the business when status is wrong-URL or matched.",
    )
    url_verification_notes = models.JSONField(
        default=dict,
        blank=True,
        help_text="Debug payload: dns_ok, http_ok, status_code, error, etc.",
    )
    verified_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When live/broken verification last ran (wrong-URL path only).",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO extraction snapshot"
        verbose_name_plural = "AEO extraction snapshots"

    def __str__(self) -> str:
        return f"AEOExtractionSnapshot(response_id={self.response_snapshot_id}, ok={not self.extraction_parse_failed})"


class AEOPromptExecutionAggregate(models.Model):
    """
    Canonical per-prompt aggregate (read model) updated in-place across passes/providers.

    Raw attempt artifacts still live in AEOResponseSnapshot/AEOExtractionSnapshot.
    """

    STABILITY_PENDING = "pending"
    STABILITY_STABLE = "stable"
    STABILITY_UNSTABLE = "unstable"
    STABILITY_STABILIZED_AFTER_THIRD = "stabilized_after_third"
    STABILITY_CHOICES = [
        (STABILITY_PENDING, "Pending"),
        (STABILITY_STABLE, "Stable"),
        (STABILITY_UNSTABLE, "Unstable"),
        (STABILITY_STABILIZED_AFTER_THIRD, "Stabilized After Third"),
    ]

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_prompt_execution_aggregates",
    )
    execution_run = models.ForeignKey(
        "AEOExecutionRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="prompt_aggregates",
    )
    prompt_hash = models.CharField(max_length=64, db_index=True)
    prompt_text = models.TextField(blank=True, default="")
    prompt_type = models.CharField(max_length=32, blank=True, default="")
    prompt_category = models.CharField(max_length=32, blank=True, default="")
    is_custom_prompt = models.BooleanField(
        default=False,
        help_text="True when this aggregate row tracks a user-added custom monitored prompt.",
    )

    openai_pass_count = models.IntegerField(default=0)
    gemini_pass_count = models.IntegerField(default=0)
    openai_brand_cited_count = models.IntegerField(default=0)
    gemini_brand_cited_count = models.IntegerField(default=0)
    openai_wrong_url_count = models.IntegerField(default=0)
    gemini_wrong_url_count = models.IntegerField(default=0)
    total_pass_count = models.IntegerField(default=0)
    total_brand_cited_count = models.IntegerField(default=0)

    last_openai_response_snapshot = models.ForeignKey(
        AEOResponseSnapshot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    last_gemini_response_snapshot = models.ForeignKey(
        AEOResponseSnapshot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    last_openai_competitors_json = models.JSONField(default=list, blank=True)
    last_gemini_competitors_json = models.JSONField(default=list, blank=True)
    last_openai_citations_json = models.JSONField(default=list, blank=True)
    last_gemini_citations_json = models.JSONField(default=list, blank=True)
    last_openai_brand_mentioned = models.BooleanField(default=False)
    last_gemini_brand_mentioned = models.BooleanField(default=False)
    openai_last_wrong_url_status = models.CharField(max_length=40, blank=True, default="")
    gemini_last_wrong_url_status = models.CharField(max_length=40, blank=True, default="")
    openai_brand_mention_history = models.JSONField(default=list, blank=True)
    gemini_brand_mention_history = models.JSONField(default=list, blank=True)
    openai_pass_history_json = models.JSONField(default=list, blank=True)
    gemini_pass_history_json = models.JSONField(default=list, blank=True)
    combined_competitor_counts = models.JSONField(default=dict, blank=True)
    combined_citation_counts = models.JSONField(default=dict, blank=True)
    combined_provider_breakdown = models.JSONField(default=dict, blank=True)
    combined_total_passes_observed = models.IntegerField(default=0)
    combined_total_unique_competitors = models.IntegerField(default=0)
    combined_total_unique_citations = models.IntegerField(default=0)
    combined_last_recomputed_at = models.DateTimeField(null=True, blank=True)
    openai_stability_status = models.CharField(
        max_length=28,
        choices=STABILITY_CHOICES,
        default=STABILITY_PENDING,
        db_index=True,
    )
    gemini_stability_status = models.CharField(
        max_length=28,
        choices=STABILITY_CHOICES,
        default=STABILITY_PENDING,
        db_index=True,
    )
    openai_third_pass_required = models.BooleanField(default=False)
    gemini_third_pass_required = models.BooleanField(default=False)
    openai_third_pass_ran = models.BooleanField(default=False)
    gemini_third_pass_ran = models.BooleanField(default=False)

    perplexity_pass_count = models.IntegerField(default=0)
    perplexity_brand_cited_count = models.IntegerField(default=0)
    perplexity_wrong_url_count = models.IntegerField(default=0)
    last_perplexity_response_snapshot = models.ForeignKey(
        AEOResponseSnapshot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    last_perplexity_competitors_json = models.JSONField(default=list, blank=True)
    last_perplexity_citations_json = models.JSONField(default=list, blank=True)
    last_perplexity_brand_mentioned = models.BooleanField(default=False)
    perplexity_last_wrong_url_status = models.CharField(max_length=40, blank=True, default="")
    perplexity_brand_mention_history = models.JSONField(default=list, blank=True)
    perplexity_pass_history_json = models.JSONField(default=list, blank=True)
    perplexity_stability_status = models.CharField(
        max_length=28,
        choices=STABILITY_CHOICES,
        default=STABILITY_PENDING,
        db_index=True,
    )
    perplexity_third_pass_required = models.BooleanField(default=False)
    perplexity_third_pass_ran = models.BooleanField(default=False)

    stability_status = models.CharField(
        max_length=28,
        choices=STABILITY_CHOICES,
        default=STABILITY_PENDING,
        db_index=True,
    )
    stability_reasons = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at", "-id")
        constraints = [
            models.UniqueConstraint(
                fields=["profile", "execution_run", "prompt_hash"],
                name="accounts_aeo_prompt_agg_profile_run_hash_uq",
            ),
        ]
        indexes = [
            models.Index(
                fields=["profile", "execution_run", "stability_status"],
                # <= 30 chars for SQLite / Django system checks (E034)
                name="acct_aeoagg_runstat_idx",
            ),
            models.Index(
                fields=["profile", "prompt_hash"],
                name="acct_aeoagg_prhash_idx",
            ),
        ]

    def __str__(self) -> str:
        return f"AEOPromptExecutionAggregate(profile_id={self.profile_id}, hash={self.prompt_hash[:12]}...)"


class AEOScoreSnapshot(models.Model):
    """
    Append-only AEO scoring run derived from extraction snapshots (trend / reporting).

    New rows are created for each scoring pass; historical rows are never updated in place.
    """

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_score_snapshots",
    )
    LAYER_SAMPLE = "sample"
    LAYER_CONFIDENCE = "confidence"
    SCORE_LAYER_CHOICES = [
        (LAYER_SAMPLE, "Sample"),
        (LAYER_CONFIDENCE, "Confidence"),
    ]
    score_layer = models.CharField(
        max_length=16,
        choices=SCORE_LAYER_CHOICES,
        default=LAYER_CONFIDENCE,
        db_index=True,
    )
    execution_run = models.ForeignKey(
        "AEOExecutionRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="score_snapshots",
    )
    visibility_score = models.FloatField(default=0.0)
    weighted_position_score = models.FloatField(default=0.0)
    citation_share = models.FloatField(default=0.0)
    competitor_dominance_json = models.JSONField(default=dict, blank=True)
    total_prompts = models.IntegerField(default=0)
    total_mentions = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO score snapshot"
        verbose_name_plural = "AEO score snapshots"

    def __str__(self) -> str:
        return f"AEOScoreSnapshot(profile_id={self.profile_id}, prompts={self.total_prompts})"


class AEORecommendationRun(models.Model):
    """
    Append-only recommendation pass from Phase 4 scores + Phase 3 extractions.

    Tracks completed actions / effectiveness in JSON for future workflows (not enforced here).
    """

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_recommendation_runs",
    )
    score_snapshot = models.ForeignKey(
        AEOScoreSnapshot,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recommendation_runs",
    )
    recommendations_json = models.JSONField(default=list, blank=True)
    strategies_json = models.JSONField(
        default=list,
        blank=True,
        help_text="UI-ready hierarchical strategies (grouped by parent_group_id, deduped actions).",
    )
    visibility_score_at_run = models.FloatField(default=0.0)
    weighted_position_score_at_run = models.FloatField(default=0.0)
    citation_share_at_run = models.FloatField(default=0.0)
    actions_completed_json = models.JSONField(
        default=list,
        blank=True,
        help_text="Optional log of completed actions (indices, timestamps) for trend tracking.",
    )
    effectiveness_json = models.JSONField(
        default=dict,
        blank=True,
        help_text="Placeholder for future outcome / effectiveness metrics per recommendation.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO recommendation run"
        verbose_name_plural = "AEO recommendation runs"

    def __str__(self) -> str:
        recs = self.recommendations_json
        n = len(recs) if isinstance(recs, list) else 0
        return f"AEORecommendationRun(profile_id={self.profile_id}, items={n})"


class AEOExecutionRun(models.Model):
    """
    Phase 1 AEO execution run tracker for prompt batch processing.
    """

    FETCH_MODE_CACHE_HIT = "cache_hit"
    FETCH_MODE_FRESH_FETCH = "fresh_fetch"
    FETCH_MODE_CHOICES = [
        (FETCH_MODE_CACHE_HIT, "Cache Hit"),
        (FETCH_MODE_FRESH_FETCH, "Fresh Fetch"),
    ]

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_SKIPPED_CACHED = "skipped_cached"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
        (STATUS_SKIPPED_CACHED, "Skipped Cached"),
    ]
    STAGE_PENDING = "pending"
    STAGE_RUNNING = "running"
    STAGE_COMPLETED = "completed"
    STAGE_FAILED = "failed"
    STAGE_SKIPPED = "skipped"
    STAGE_CHOICES = [
        (STAGE_PENDING, "Pending"),
        (STAGE_RUNNING, "Running"),
        (STAGE_COMPLETED, "Completed"),
        (STAGE_FAILED, "Failed"),
        (STAGE_SKIPPED, "Skipped"),
    ]

    profile = models.ForeignKey(
        BusinessProfile,
        on_delete=models.CASCADE,
        related_name="aeo_execution_runs",
    )
    prompt_count_requested = models.IntegerField(default=0)
    prompt_count_executed = models.IntegerField(default=0)
    prompt_count_failed = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    cache_hit = models.BooleanField(default=False)
    fetch_mode = models.CharField(
        max_length=16,
        choices=FETCH_MODE_CHOICES,
        default=FETCH_MODE_FRESH_FETCH,
    )
    status = models.CharField(
        max_length=24,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    extraction_status = models.CharField(
        max_length=16,
        choices=STAGE_CHOICES,
        default=STAGE_PENDING,
    )
    scoring_status = models.CharField(
        max_length=16,
        choices=STAGE_CHOICES,
        default=STAGE_PENDING,
    )
    recommendation_status = models.CharField(
        max_length=16,
        choices=STAGE_CHOICES,
        default=STAGE_PENDING,
    )
    background_status = models.CharField(
        max_length=16,
        choices=STAGE_CHOICES,
        default=STAGE_PENDING,
    )
    phase1_completed_at = models.DateTimeField(null=True, blank=True)
    phase1_provider_calls = models.IntegerField(default=0)
    extraction_count = models.IntegerField(default=0)
    score_snapshot_id = models.IntegerField(null=True, blank=True)
    recommendation_run_id = models.IntegerField(null=True, blank=True)
    seo_triggered_at = models.DateTimeField(null=True, blank=True)
    seo_trigger_status = models.CharField(max_length=32, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO execution run"
        verbose_name_plural = "AEO execution runs"

    def __str__(self) -> str:
        return f"AEOExecutionRun(profile_id={self.profile_id}, status={self.status})"


class OnboardingOnPageCrawl(models.Model):
    """
    Stores DataForSEO On-Page crawl results during onboarding (up to 10 pages).
    """

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
    ]
    PROMPT_PLAN_PENDING = "pending"
    PROMPT_PLAN_QUEUED = "queued"
    PROMPT_PLAN_RUNNING = "running"
    PROMPT_PLAN_COMPLETED = "completed"
    PROMPT_PLAN_FAILED = "failed"
    PROMPT_PLAN_STATUS_CHOICES = [
        (PROMPT_PLAN_PENDING, "Pending"),
        (PROMPT_PLAN_QUEUED, "Queued"),
        (PROMPT_PLAN_RUNNING, "Running"),
        (PROMPT_PLAN_COMPLETED, "Completed"),
        (PROMPT_PLAN_FAILED, "Failed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="onboarding_onpage_crawls",
    )
    business_profile = models.ForeignKey(
        "BusinessProfile",
        on_delete=models.CASCADE,
        related_name="onboarding_onpage_crawls",
    )
    domain = models.CharField(max_length=255)
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default=STATUS_PENDING)
    max_pages = models.PositiveSmallIntegerField(default=10)
    pages = models.JSONField(default=list, blank=True)
    # DataForSEO Labs ranked_keywords/live (domain) + topic clustering vs on-page seeds
    ranked_keywords = models.JSONField(
        default=list,
        blank=True,
        help_text="Normalized ranked keyword rows from Labs (keyword, rank, volume).",
    )
    topic_clusters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Topic clusters: crawl seeds matched to ranked keywords (clusters, unclustered, stats).",
    )
    crawl_topic_seeds = models.JSONField(
        default=list,
        blank=True,
        help_text="Service/topic phrases extracted from the on-page crawl only.",
    )
    # Some deployments added this column for an earlier onboarding experiment; ORM must populate it (default []).
    business_topics = models.JSONField(
        default=list,
        blank=True,
        help_text="Unused; onboarding review topics use ``review_topics``. Kept for database compatibility.",
    )
    ranked_keywords_error = models.TextField(
        blank=True,
        default="",
        help_text="Set when Labs ranked_keywords call fails (crawl may still succeed).",
    )
    # Labs ranked_keywords fetch runs in a separate Celery task after review_topics are ready.
    RANKED_FETCH_LEGACY = ""
    RANKED_FETCH_PENDING = "pending"
    RANKED_FETCH_COMPLETE = "complete"
    RANKED_FETCH_STATUS_CHOICES = [
        (RANKED_FETCH_LEGACY, "Legacy / not applicable"),
        (RANKED_FETCH_PENDING, "Pending background fetch"),
        (RANKED_FETCH_COMPLETE, "Complete"),
    ]
    ranked_keywords_fetch_status = models.CharField(
        max_length=16,
        blank=True,
        default="",
        choices=RANKED_FETCH_STATUS_CHOICES,
        help_text="pending=background Labs fetch in flight; complete=done or skipped; empty=pre-async rows.",
    )
    # Gemini (or future LLM) review topics for onboarding Step 2 — independent of ranked_keywords.
    review_topics = models.JSONField(
        default=list,
        blank=True,
        help_text='LLM-generated business topics, e.g. [{"topic": "...", "category": "...", "rationale": "..."}].',
    )
    review_topics_error = models.TextField(
        blank=True,
        default="",
        help_text="Set when review topic generation fails or returns nothing usable.",
    )
    task_id = models.CharField(max_length=128, blank=True, default="")
    exit_reason = models.CharField(max_length=64, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    prompt_plan_status = models.CharField(
        max_length=16,
        choices=PROMPT_PLAN_STATUS_CHOICES,
        default=PROMPT_PLAN_PENDING,
        db_index=True,
    )
    prompt_plan_prompt_count = models.IntegerField(default=0)
    prompt_plan_error = models.TextField(blank=True, default="")
    prompt_plan_task_id = models.CharField(max_length=128, blank=True, default="")
    prompt_plan_started_at = models.DateTimeField(null=True, blank=True)
    prompt_plan_finished_at = models.DateTimeField(null=True, blank=True)
    context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional business_name, location for mention detection.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "Onboarding on-page crawl"
        verbose_name_plural = "Onboarding on-page crawls"

    def __str__(self) -> str:
        return f"OnboardingOnPageCrawl(profile_id={self.business_profile_id}, {self.status})"




class AgentConversation(models.Model):
    """
    Generic conversation for an agent (e.g. SEO, Ads, Reviews).
    """

    AGENT_CHOICES = [
        ("seo", "SEO Agent"),
        ("ads", "Ads Agent"),
        ("reviews", "Reviews Agent"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_conversations",
    )
    agent = models.CharField(max_length=32, choices=AGENT_CHOICES)
    title = models.CharField(max_length=255, blank=True)
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"AgentConversation(user={self.user!s}, agent={self.agent})"


class AgentMessage(models.Model):
    """
    Individual message within an agent conversation.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]

    conversation = models.ForeignKey(
        AgentConversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"AgentMessage(conv={self.conversation_id}, role={self.role})"


class AgentActivityLog(models.Model):
    """
    Log of what each agent did for the dashboard "What your agents did today".
    Cleaned automatically: records older than 30 days are removed (see management command).
    """

    AGENT_CHOICES = [
        ("seo", "SEO Agent"),
        ("ads", "Ads Agent"),
        ("reviews", "Reviews Agent"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_activity_logs",
    )
    agent = models.CharField(max_length=32, choices=AGENT_CHOICES)
    description = models.TextField(help_text="What was completed")
    account_name = models.CharField(
        max_length=128,
        blank=True,
        help_text="Optional: connected account/integration (e.g. Google Ads, Google Search Console)",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Agent activity log"
        verbose_name_plural = "Agent activity logs"

    def __str__(self) -> str:
        return f"AgentActivityLog(user={self.user!s}, agent={self.agent}, at={self.created_at})"


class ThirdPartyApiProvider(models.TextChoices):
    """Shared provider slug for usage logs and error logs."""

    DATAFORSEO = "dataforseo", "DataForSEO"
    OPENAI = "openai", "OpenAI"
    GEMINI = "gemini", "Google Gemini"
    PERPLEXITY = "perplexity", "Perplexity"


class ThirdPartyApiRequestLog(models.Model):
    """
    One row per outbound call to DataForSEO, OpenAI, Gemini, or Perplexity (for usage graphs and cost tracking).
    """

    Provider = ThirdPartyApiProvider  # backward-compatible alias

    provider = models.CharField(max_length=32, choices=ThirdPartyApiProvider.choices, db_index=True)
    business_profile = models.ForeignKey(
        BusinessProfile,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="third_party_api_logs",
    )
    operation = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="Endpoint or logical operation name (e.g. DataForSEO path, openai.chat operation).",
    )
    cost_usd = models.DecimalField(
        max_digits=14,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Provider-reported or estimated cost in USD.",
    )
    tokens_sent = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Input/prompt tokens (OpenAI/Gemini); null for providers without token usage.",
    )
    tokens_received = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Output/completion tokens (OpenAI/Gemini); null for providers without token usage.",
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Third-party API request"
        verbose_name_plural = "Third-party API requests"
        indexes = [
            models.Index(fields=["provider", "created_at"]),
            models.Index(fields=["business_profile", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.provider} {self.operation[:40]!r} @ {self.created_at}"


class ThirdPartyApiErrorLog(models.Model):
    """
    Failed or erroneous outbound third-party API calls (debugging; separate from usage/cost rows).
    """

    class ErrorKind(models.TextChoices):
        HTTP_ERROR = "http_error", "HTTP error"
        TIMEOUT = "timeout", "Timeout"
        CONNECTION_ERROR = "connection_error", "Connection error"
        PARSE_ERROR = "parse_error", "Parse error"
        VALIDATION_ERROR = "validation_error", "Validation error"
        UNKNOWN_EXCEPTION = "unknown_exception", "Unknown exception"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    provider = models.CharField(max_length=32, choices=ThirdPartyApiProvider.choices, db_index=True)
    business_profile = models.ForeignKey(
        BusinessProfile,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="third_party_api_error_logs",
    )
    operation = models.CharField(
        max_length=512,
        blank=True,
        default="",
        help_text="Endpoint or logical operation name (same convention as ThirdPartyApiRequestLog).",
    )
    http_status = models.PositiveSmallIntegerField(null=True, blank=True)
    error_kind = models.CharField(
        max_length=32,
        choices=ErrorKind.choices,
        default=ErrorKind.UNKNOWN_EXCEPTION,
        db_index=True,
    )
    message = models.CharField(
        max_length=1024,
        blank=True,
        default="",
        help_text="Short summary (exception message or status-derived text).",
    )
    detail = models.TextField(
        blank=True,
        default="",
        help_text="Truncated response body or extra context (no full prompts).",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Third-party API error"
        verbose_name_plural = "Third-party API errors"
        indexes = [
            models.Index(fields=["provider", "created_at"]),
            models.Index(fields=["business_profile", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.provider} {self.error_kind} {self.operation[:32]!r} @ {self.created_at}"


