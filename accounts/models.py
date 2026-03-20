from django.conf import settings
from django.db import models


class BusinessProfile(models.Model):
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
    top_keywords = models.JSONField(default=list, blank=True)
    total_search_volume = models.IntegerField(default=0)
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
        unique_together = ("user", "period_start")
        verbose_name = "SEO overview snapshot"
        verbose_name_plural = "SEO overview snapshots"

    def __str__(self) -> str:
        return f"SEOOverviewSnapshot(user={self.user!s}, period_start={self.period_start})"


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


