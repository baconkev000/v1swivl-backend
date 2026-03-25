from datetime import timedelta

# Central cache/snapshot TTL settings for accounts/SEO/AEO flows.

# Main SEO/AEO snapshot cadence.
SEO_SNAPSHOT_TTL = timedelta(days=7)
AEO_SNAPSHOT_TTL = timedelta(days=30)

# Async enrichment TTLs.
SEO_KEYWORDS_ENRICHMENT_TTL = timedelta(days=7)
SEO_NEXT_STEPS_TTL = timedelta(days=7)
SEO_KEYWORD_ACTION_TTL = timedelta(days=7)
AEO_RECOMMENDATIONS_TTL = timedelta(days=7)

# API cache TTLs.
SEO_KEYWORDS_API_CACHE_TTL = timedelta(days=7)
SEO_LOCATIONS_SEARCH_CACHE_TTL = timedelta(days=1)

# Non-SEO third-party API cache cadence.
THIRD_PARTY_API_CACHE_TTL = timedelta(hours=1)
