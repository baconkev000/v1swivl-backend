from django.conf import settings
from django.db import models


class BusinessProfile(models.Model):
    """
    Stores business profile settings for a user.

    Each user has at most one BusinessProfile, linked via a foreign key
    to the Django user model (which includes the user's email).
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="business_profile",
    )

    full_name = models.CharField(max_length=255, blank=True)
    business_name = models.CharField(max_length=255, blank=True)
    business_address = models.CharField(max_length=255, blank=True)

    # Industry is intentionally a free-text field (no choices).
    industry = models.CharField(max_length=255, blank=True)

    # Tone of voice for marketing/communications.
    tone_of_voice = models.CharField(max_length=64, blank=True)

    phone = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    website_url = models.URLField(blank=True)

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


class SEOOverviewSnapshot(models.Model):
    """
    Stores monthly SEO overview metrics for a user
    (first day of the month + last time data was fetched).
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

    class Meta:
        unique_together = ("user", "period_start")
        verbose_name = "SEO overview snapshot"
        verbose_name_plural = "SEO overview snapshots"

    def __str__(self) -> str:
        return f"SEOOverviewSnapshot(user={self.user!s}, period_start={self.period_start})"


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



