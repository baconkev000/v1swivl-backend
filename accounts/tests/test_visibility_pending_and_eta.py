"""Visibility pending breakdown, prompt-coverage repair hook, and full-phase ETA edge cases."""

import pytest
from django.contrib.auth import get_user_model

from accounts.aeo.prompt_full_ready import compute_full_phase_eta_seconds
from accounts.aeo.visibility_pending import aeo_visibility_pending_breakdown
from accounts.models import BusinessProfile

User = get_user_model()


def test_compute_full_phase_eta_zero_remaining_not_cold_start():
    """remaining==0 ⇒ eta 0 and not cold — callers must not show a minutes estimate from this."""
    eta_sec, cold = compute_full_phase_eta_seconds([120.0, 180.0], remaining=0, full_phase_completed=5)
    assert eta_sec == 0
    assert cold is False


def test_compute_full_phase_eta_positive_remaining_uses_durations():
    eta_sec, cold = compute_full_phase_eta_seconds([60.0, 60.0], remaining=2, full_phase_completed=1)
    assert cold is False
    assert eta_sec is not None
    assert eta_sec > 0


@pytest.mark.django_db
def test_visibility_pending_breakdown_empty_monitored():
    user = User.objects.create_user(username="v1", email="v1@example.com", password="pw")
    profile = BusinessProfile.objects.create(user=user, is_main=True, business_name="B", selected_aeo_prompts=[])
    bd = aeo_visibility_pending_breakdown(profile)
    assert bd["visibility_pending"] is False
    assert bd["execution_inflight"] is False
    assert bd["latest_run_extractions_inflight"] is False
    assert bd["snapshots_awaiting_extraction"] is False


