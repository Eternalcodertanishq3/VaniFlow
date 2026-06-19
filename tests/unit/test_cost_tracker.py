"""
Unit tests for Token & Cost Optimization Tracker.
"""
import pytest
from vaaniflow.cost import CostTracker


@pytest.fixture
def tracker():
    """Fresh cost tracker for each test."""
    t = CostTracker.__new__(CostTracker)
    t._initialized = False
    t.__init__()
    t.reset()
    return t


def test_initial_snapshot_zeroes(tracker):
    """Fresh tracker should report all zeros."""
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["translation_api_calls_made"] == 0
    assert snap["cost_optimization"]["translation_api_calls_avoided_via_cache"] == 0
    assert snap["cost_optimization"]["cache_hit_rate_pct"] == 0.0
    assert snap["operations"]["total_jobs_completed"] == 0


def test_record_cache_hit(tracker):
    """Cache hits should be counted and affect hit rate."""
    tracker.record_cache_hit(5)
    tracker.record_translation_call("sarvam", 5)
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["translation_api_calls_avoided_via_cache"] == 5
    assert snap["cost_optimization"]["cache_hit_rate_pct"] == 50.0


def test_record_translation_calls(tracker):
    """API calls should be tracked per provider."""
    tracker.record_translation_call("sarvam", 10)
    tracker.record_translation_call("google", 3)
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["translation_api_calls_made"] == 13
    assert snap["provider_call_breakdown"]["sarvam"] == 10
    assert snap["provider_call_breakdown"]["google"] == 3


def test_record_tts_calls(tracker):
    """TTS calls should be tracked with provider-specific cost estimates."""
    tracker.record_tts_call("sarvam", 20)
    snap = tracker.get_snapshot()
    assert snap["tts_costs"]["sarvam"]["calls"] == 20
    assert snap["tts_costs"]["sarvam"]["estimated_cost_usd"] > 0


def test_cost_savings_calculation(tracker):
    """Savings should be calculated based on calls avoided."""
    tracker.record_cache_hit(100)
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["estimated_translation_savings_usd"] > 0


def test_record_job_completed(tracker):
    """Job completion counter should increment."""
    tracker.record_job_completed()
    tracker.record_job_completed()
    snap = tracker.get_snapshot()
    assert snap["operations"]["total_jobs_completed"] == 2


def test_record_segments(tracker):
    """Segment counter should accumulate."""
    tracker.record_segments(15)
    tracker.record_segments(10)
    snap = tracker.get_snapshot()
    assert snap["operations"]["total_segments_processed"] == 25


def test_100_pct_cache_hit_rate(tracker):
    """All cache hits, zero API calls = 100% rate."""
    tracker.record_cache_hit(50)
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["cache_hit_rate_pct"] == 100.0


def test_reset(tracker):
    """Reset should clear all counters."""
    tracker.record_translation_call("sarvam", 10)
    tracker.record_cache_hit(5)
    tracker.record_tts_call("sarvam", 20)
    tracker.record_job_completed()
    tracker.reset()
    snap = tracker.get_snapshot()
    assert snap["cost_optimization"]["translation_api_calls_made"] == 0
    assert snap["operations"]["total_jobs_completed"] == 0


def test_uptime_is_positive(tracker):
    """Uptime should be a positive number."""
    snap = tracker.get_snapshot()
    assert snap["operations"]["uptime_seconds"] >= 0
