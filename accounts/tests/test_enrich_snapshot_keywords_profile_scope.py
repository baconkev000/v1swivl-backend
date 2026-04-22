"""
Regression: keyword enrichment must use the snapshot's business_profile, not is_main.
"""

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone as dj_tz

from accounts.models import BusinessProfile, SEOOverviewSnapshot

User = get_user_model()


@pytest.mark.django_db
def test_enrich_snapshot_keywords_task_passes_snapshots_business_profile_to_labs(monkeypatch):
    from accounts import tasks as accounts_tasks

    user = User.objects.create_user(
        username="enrich_scope",
        email="enrich_scope@example.com",
        password="pw",
    )
    main = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Main Co",
        website_url="https://main.example",
        business_address="US",
    )
    secondary = BusinessProfile.objects.create(
        user=user,
        is_main=False,
        business_name="Second Co",
        website_url="https://second.example",
        business_address="US",
    )
    period_start = dj_tz.now().date().replace(day=1)
    top_keywords = [{"keyword": f"kw{i}", "rank": 5 + i} for i in range(20)]
    snap = SEOOverviewSnapshot.objects.create(
        user=user,
        business_profile=secondary,
        period_start=period_start,
        cached_domain="second.example",
        cached_location_mode="organic",
        cached_location_code=0,
        top_keywords=top_keywords,
        keywords_enriched_at=None,
    )

    bp_seen: list[int | None] = []
    gap_bp_seen: list[int | None] = []
    llm_bp_seen: list[int | None] = []

    def _stub_gap(domain, location_code, language_code, user, top_keywords, **kwargs):
        gap_bp_seen.append(getattr(kwargs.get("business_profile"), "pk", None))
        return None

    def _stub_llm(user, location_code, top_keywords, **kwargs):
        llm_bp_seen.append(getattr(kwargs.get("business_profile"), "pk", None))
        return None

    def _stub_rank(
        *,
        domain,
        location_code,
        language_code,
        top_keywords,
        user,
        business_profile=None,
        **kwargs,
    ):
        bp_seen.append(getattr(business_profile, "pk", None))
        return {
            "total": 100,
            "non_null_after": 100,
            "filled_from_ranked": 0,
            "filled_from_gap": 0,
        }

    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_gap_keywords", _stub_gap)
    monkeypatch.setattr("accounts.dataforseo_utils.enrich_with_llm_keywords", _stub_llm)
    monkeypatch.setattr("accounts.dataforseo_utils.enrich_keyword_ranks_from_labs", _stub_rank)

    def _noop_delay(*args, **kwargs):
        return None

    monkeypatch.setattr(accounts_tasks.generate_snapshot_next_steps_task, "delay", _noop_delay)
    monkeypatch.setattr(accounts_tasks.generate_keyword_action_suggestions_task, "delay", _noop_delay)

    accounts_tasks.enrich_snapshot_keywords_task.apply(args=(snap.pk,))

    assert bp_seen == [secondary.pk]
    assert gap_bp_seen == [secondary.pk]
    assert llm_bp_seen == [secondary.pk]
    assert main.is_main is True and secondary.is_main is False
