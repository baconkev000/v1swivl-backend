from django.conf import settings
from django.db import models


class BusinessProfile(models.Model):
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

    # Industry is intentionally a free-text field (no choices).
    industry = models.CharField(max_length=255, blank=True)

    # Plan type for this business (free, pro, scale, etc.).
    plan = models.CharField(max_length=64, blank=True)

    # Tone of voice for marketing/communications.
    tone_of_voice = models.CharField(max_length=64, blank=True)

    phone = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    website_url = models.URLField(blank=True)

    # Whether this is the main / primary profile for the user.
    is_main = models.BooleanField(default=False)

    # Optional: comma-separated competitor domains for SEO keyword gap analysis.
    # When provided, these are preferred over DataForSEO auto-competitors.
    # Example: "smilebright.com, nocavityclinic.com, oakdentalcare.com"
    seo_competitor_domains_override = models.TextField(blank=True, default="")
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

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Business profile"
        verbose_name_plural = "Business profiles"

    def __str__(self) -> str:
        return f"BusinessProfile(user={self.user!s})"


class GoogleSearchConsoleConnection(models.Model):
    """
    Tracks whether a user has granted this app access to Google Search Console.

    Tokens are stored to enable read-only API access (webmasters.readonly scope).
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="gsc_connection",
    )

    access_token = models.TextField(blank=True)
    refresh_token = models.TextField(blank=True)
    token_type = models.CharField(max_length=32, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Google Search Console connection"
        verbose_name_plural = "Google Search Console connections"

    def __str__(self) -> str:
        return f"GSCConnection(user={self.user!s})"


class GoogleBusinessProfileConnection(models.Model):
    """
    Tracks whether a user has granted this app access to Google Business Profile (reviews, locations).
    Used by the Reviews Agent to pull star rating, total reviews, and response rate.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="gbp_connection",
    )

    access_token = models.TextField(blank=True)
    refresh_token = models.TextField(blank=True)
    token_type = models.CharField(max_length=32, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Google Business Profile connection"
        verbose_name_plural = "Google Business Profile connections"

    def __str__(self) -> str:
        return f"GBPConnection(user={self.user!s})"


class GoogleAdsConnection(models.Model):
    """
    Tracks whether a user has granted this app access to their Google Ads account.

    Stores OAuth tokens needed to call the Google Ads API on behalf of the user.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="google_ads_connection",
    )

    access_token = models.TextField(blank=True)
    refresh_token = models.TextField(blank=True)
    token_type = models.CharField(max_length=32, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    # Optional per-user customer ID (e.g. 123-456-7890). When set, this
    # overrides the app-level GOOGLE_ADS_CUSTOMER_ID env for API calls.
    customer_id = models.CharField(max_length=32, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Google Ads connection"
        verbose_name_plural = "Google Ads connections"

    def __str__(self) -> str:
        return f"GoogleAdsConnection(user={self.user!s})"


class MetaAdsConnection(models.Model):
    """
    Tracks whether a user has granted this app access to Meta (Facebook) Marketing API.

    Stores OAuth access token needed to manage ad campaigns, ad sets, ads, creatives,
    audiences, and (optionally) page posts and insights. Token is long-lived (60 days);
    refresh before expiry to keep the connection active.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meta_ads_connection",
    )

    access_token = models.TextField(blank=True)
    token_type = models.CharField(max_length=32, blank=True, default="Bearer")
    expires_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Meta Ads connection"
        verbose_name_plural = "Meta Ads connections"

    def __str__(self) -> str:
        return f"MetaAdsConnection(user={self.user!s})"


class SEOOverviewSnapshot(models.Model):
    """
    Stores monthly SEO overview metrics for a user
    (first day of the month + last time data was fetched).
    Keyword list and search metrics are cached and only refreshed once per hour
    or when the user changes their business profile website URL (domain).
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
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

    class Meta:
        unique_together = ("user", "period_start", "cached_location_mode", "cached_location_code")
        verbose_name = "SEO overview snapshot"
        verbose_name_plural = "SEO overview snapshots"

    def __str__(self) -> str:
        return f"SEOOverviewSnapshot(user={self.user!s}, period_start={self.period_start})"

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
        constraints = [
            models.UniqueConstraint(
                fields=["execution_run", "prompt_hash", "platform"],
                condition=models.Q(execution_run__isnull=False),
                name="accounts_aeo_rsp_run_hash_platform_uq",
            ),
        ]

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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "AEO extraction snapshot"
        verbose_name_plural = "AEO extraction snapshots"

    def __str__(self) -> str:
        return f"AEOExtractionSnapshot(response_id={self.response_snapshot_id}, ok={not self.extraction_parse_failed})"


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


class OnPageAuditSnapshot(models.Model):
    """
    Cached On-Page / Technical SEO audit metrics per user & domain.
    Used to avoid running the DataForSEO On-Page crawler more than once
    every 24 hours for the same site.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="onpage_audit_snapshots",
    )
    domain = models.CharField(max_length=255)
    last_fetched_at = models.DateTimeField(auto_now=True)

    # Raw counts
    pages_missing_titles = models.IntegerField(default=0)
    pages_missing_descriptions = models.IntegerField(default=0)
    pages_bad_h1 = models.IntegerField(default=0)
    images_missing_alt = models.IntegerField(default=0)
    broken_internal_links = models.IntegerField(default=0)
    error_pages_4xx_5xx = models.IntegerField(default=0)
    pages_missing_canonical = models.IntegerField(default=0)
    duplicate_canonical_targets = models.IntegerField(default=0)
    has_robots_txt = models.BooleanField(default=False)
    has_sitemap_xml = models.BooleanField(default=False)

    # How many "important" pages were included in the last audit.
    pages_audited = models.IntegerField(default=0)

    # Scores
    metadata_score = models.IntegerField(default=0)
    content_structure_score = models.IntegerField(default=0)
    accessibility_score = models.IntegerField(default=0)
    internal_link_score = models.IntegerField(default=0)
    indexability_score = models.IntegerField(default=0)
    onpage_seo_score = models.IntegerField(default=0)
    technical_seo_score = models.IntegerField(default=0)

    # Human-readable summaries for dashboard display
    issue_summaries = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ("user", "domain")
        verbose_name = "On-page audit snapshot"
        verbose_name_plural = "On-page audit snapshots"

    def __str__(self) -> str:
        return f"OnPageAuditSnapshot(user={self.user!s}, domain={self.domain})"


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
    ranked_keywords_error = models.TextField(
        blank=True,
        default="",
        help_text="Set when Labs ranked_keywords call fails (crawl may still succeed).",
    )
    task_id = models.CharField(max_length=128, blank=True, default="")
    exit_reason = models.CharField(max_length=64, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
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


class ReviewsOverviewSnapshot(models.Model):
    """
    Cached reviews/GBP overview metrics per user (star rating, total reviews, response rate, etc.).
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reviews_overview_snapshot",
    )

    star_rating = models.DecimalField(max_digits=3, decimal_places=2, default=0, null=True, blank=True)
    previous_star_rating = models.DecimalField(max_digits=3, decimal_places=2, default=0, null=True, blank=True)
    total_reviews = models.IntegerField(default=0)
    new_reviews_this_month = models.IntegerField(default=0)
    response_rate_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0, null=True, blank=True)
    industry_avg_response_pct = models.DecimalField(max_digits=5, decimal_places=2, default=45, null=True, blank=True)
    requests_sent = models.IntegerField(default=0)
    conversion_pct = models.DecimalField(max_digits=5, decimal_places=2, default=0, null=True, blank=True)

    last_fetched_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Reviews overview snapshot"
        verbose_name_plural = "Reviews overview snapshots"

    def __str__(self) -> str:
        return f"ReviewsOverviewSnapshot(user={self.user!s})"


class GoogleAdsKeywordIdea(models.Model):
    """
    Cached Google Ads keyword idea metrics per user & keyword.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="google_ads_keyword_ideas",
    )
    keyword = models.CharField(max_length=255)

    avg_monthly_searches = models.IntegerField(default=0)
    competition = models.CharField(max_length=32, blank=True)
    competition_index = models.IntegerField(default=0)
    low_top_of_page_bid_micros = models.BigIntegerField(default=0)
    high_top_of_page_bid_micros = models.BigIntegerField(default=0)

    last_fetched_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "keyword")
        verbose_name = "Google Ads keyword idea"
        verbose_name_plural = "Google Ads keyword ideas"

    def __str__(self) -> str:
        return f"GoogleAdsKeywordIdea(user={self.user!s}, keyword='{self.keyword}')"


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


class ReviewsConversation(models.Model):
    """
    Conversation thread for the Reviews Agent chat. Kept separate from SEO/Ads.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reviews_conversations",
    )
    title = models.CharField(max_length=255, blank=True)
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Reviews conversation"
        verbose_name_plural = "Reviews conversations"

    def __str__(self) -> str:
        return f"ReviewsConversation(user={self.user!s}, id={self.id})"


class ReviewsMessage(models.Model):
    """
    Individual message within a Reviews Agent conversation.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]

    conversation = models.ForeignKey(
        ReviewsConversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Reviews message"
        verbose_name_plural = "Reviews messages"

    def __str__(self) -> str:
        return f"ReviewsMessage(conv={self.conversation_id}, role={self.role})"


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


class ThirdPartyApiRequestLog(models.Model):
    """
    One row per outbound call to DataForSEO, OpenAI, or Gemini (for usage graphs and cost tracking).
    """

    class Provider(models.TextChoices):
        DATAFORSEO = "dataforseo", "DataForSEO"
        OPENAI = "openai", "OpenAI"
        GEMINI = "gemini", "Google Gemini"

    provider = models.CharField(max_length=32, choices=Provider.choices, db_index=True)
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


