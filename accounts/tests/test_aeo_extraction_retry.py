"""Staff retry helpers for Phase-3 extractions missing on an execution run."""

import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.extraction_retry import list_aeo_response_snapshot_ids_missing_extractions
from accounts.models import (
    AEOExecutionRun,
    AEOExtractionSnapshot,
    AEOResponseSnapshot,
    BusinessProfile,
)


@pytest.mark.django_db
def test_list_missing_extractions_respects_run_and_platform():
    User = get_user_model()
    user = User.objects.create_user(username="retry_t", email="retry_t@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="RetryCo",
        plan=BusinessProfile.PLAN_PRO,
    )
    run = AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        prompt_count_requested=2,
    )
    r_openai = AEOResponseSnapshot.objects.create(
        profile=profile,
        execution_run=run,
        prompt_text="p1",
        prompt_hash="h1",
        raw_response="{}",
        platform="openai",
    )
    r_perp = AEOResponseSnapshot.objects.create(
        profile=profile,
        execution_run=run,
        prompt_text="p2",
        prompt_hash="h2",
        raw_response="{}",
        platform="perplexity",
    )
    AEOResponseSnapshot.objects.create(
        profile=profile,
        execution_run=run,
        prompt_text="p3",
        prompt_hash="h3",
        raw_response="{}",
        platform="gemini",
    )
    AEOExtractionSnapshot.objects.create(response_snapshot=r_openai, brand_mentioned=False)

    all_missing = list_aeo_response_snapshot_ids_missing_extractions(run.id)
    assert set(all_missing) == {r_perp.id, AEOResponseSnapshot.objects.get(prompt_hash="h3").id}

    perp_only = list_aeo_response_snapshot_ids_missing_extractions(run.id, platform="perplexity")
    assert perp_only == [r_perp.id]
