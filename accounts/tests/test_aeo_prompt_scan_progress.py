"""prompt_scan_* / visibility_pending helpers and API payload (when stripe available for views import)."""

import pytest

from accounts.aeo.prompt_scan_progress import monitored_prompt_keys_in_order, prompt_scan_completed_count


class _FakeExtractionRelation:
    def __init__(self, has: bool) -> None:
        self._has = has

    def exists(self) -> bool:
        return self._has


class _FakeResp:
    def __init__(self, platform: str, has_extraction: bool) -> None:
        self.platform = platform
        self.extraction_snapshots = _FakeExtractionRelation(has_extraction)


def _latest_per_platform(rows: list) -> dict:
    best: dict[str, object] = {}
    for r in rows:
        p = str(getattr(r, "platform", "") or "").strip().lower()
        if not p or p in best:
            continue
        best[p] = r
    return best


def test_monitored_prompt_keys_dedupes_and_strips():
    assert monitored_prompt_keys_in_order(["a", " a ", "b", "", "b"]) == ["a", "b"]


def test_prompt_scan_completed_zero_until_all_three_platforms_have_extractions():
    by_prompt = {
        "only": [
            _FakeResp("openai", True),
            _FakeResp("gemini", True),
        ],
    }
    assert (
        prompt_scan_completed_count(["only"], by_prompt, _latest_per_platform) == 0
    )
    by_prompt["only"].append(_FakeResp("perplexity", True))
    assert prompt_scan_completed_count(["only"], by_prompt, _latest_per_platform) == 1


def test_prompt_scan_completed_two_prompts_partial():
    by_prompt = {
        "a": [_FakeResp("openai", True), _FakeResp("gemini", True), _FakeResp("perplexity", True)],
        "b": [_FakeResp("openai", True)],
    }
    assert prompt_scan_completed_count(["a", "b"], by_prompt, _latest_per_platform) == 1


@pytest.mark.django_db
def test_prompt_coverage_api_includes_scan_fields():
    """Full HTTP test — requires ``stripe`` because ``accounts.views`` imports it."""
    pytest.importorskip("stripe")
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from accounts.models import AEOResponseSnapshot, AEOExtractionSnapshot, BusinessProfile

    User = get_user_model()
    user = User.objects.create_user(username="scan_api", email="scan_api@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["One"],
    )
    rsp = AEOResponseSnapshot.objects.create(
        profile=profile,
        prompt_text="One",
        prompt_hash="h1",
        raw_response="x",
        platform="openai",
    )
    AEOExtractionSnapshot.objects.create(response_snapshot=rsp, brand_mentioned=False)

    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/aeo/prompt-coverage/")
    assert res.status_code == 200
    data = res.json()
    assert data["prompt_scan_total"] == 1
    assert data["prompt_scan_completed"] == 0
    # Only OpenAI has a snapshot and it already has an extraction — no in-flight work.
    assert data["visibility_pending"] is False
    assert data.get("recommendations_pending") is False
    assert data["prompt_fill_completed"] == 1
    assert data["prompt_fill_target"] == 10
    assert "aeo_prompt_expansion_status" in data
    assert "aeo_prompt_expansion_last_error" in data
    assert "visibility_pending_reasons" in data
    assert data["visibility_pending_reasons"]["execution_inflight"] is False
    assert "visibility_repair" in data


@pytest.mark.django_db
def test_prompt_coverage_recommendations_pending_when_phase5_in_flight(settings):
    pytest.importorskip("stripe")
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from accounts.models import AEOExecutionRun, BusinessProfile

    settings.AEO_ENABLE_RECOMMENDATION_STAGE = True
    User = get_user_model()
    user = User.objects.create_user(username="reco_pend", email="reco_pend@example.com", password="pw")
    profile = BusinessProfile.objects.create(
        user=user,
        is_main=True,
        business_name="Biz",
        selected_aeo_prompts=["One"],
    )
    AEOExecutionRun.objects.create(
        profile=profile,
        status=AEOExecutionRun.STATUS_COMPLETED,
        scoring_status=AEOExecutionRun.STAGE_COMPLETED,
        recommendation_status=AEOExecutionRun.STAGE_RUNNING,
    )
    client = APIClient()
    client.force_authenticate(user=user)
    res = client.get("/api/aeo/prompt-coverage/")
    assert res.status_code == 200
    assert res.json().get("recommendations_pending") is True
