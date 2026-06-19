"""
Token & Cost Optimization Tracker for VaaniFlow.

Tracks API calls avoided via caching, estimated cost savings,
and provides a real-time cost dashboard via the /stats endpoint.

This is critical for enterprise Indic AI deployments where
cost-per-token at scale is a massive factor.
"""
import time
from dataclasses import dataclass, field
from threading import Lock
import structlog

log = structlog.get_logger(__name__)


@dataclass
class ProviderCostRate:
    """Cost per API call for each provider (in USD)."""
    name: str
    cost_per_translation_call: float = 0.0
    cost_per_tts_call: float = 0.0


# Estimated costs per API call (based on typical Sarvam/Google pricing)
PROVIDER_COSTS = {
    "sarvam": ProviderCostRate(
        name="sarvam",
        cost_per_translation_call=0.002,   # ~$2/1000 calls
        cost_per_tts_call=0.004,           # ~$4/1000 calls
    ),
    "google": ProviderCostRate(
        name="google",
        cost_per_translation_call=0.005,   # ~$5/1000 calls
        cost_per_tts_call=0.0,
    ),
    "elevenlabs": ProviderCostRate(
        name="elevenlabs",
        cost_per_translation_call=0.0,
        cost_per_tts_call=0.018,           # ~$18/1000 calls
    ),
    "gtts": ProviderCostRate(
        name="gtts",
        cost_per_translation_call=0.0,
        cost_per_tts_call=0.0,             # Free
    ),
}


class CostTracker:
    """
    Singleton cost tracker that accumulates API call savings across jobs.

    Thread-safe via Lock for concurrent pipeline access.
    Provides a snapshot for the /stats API endpoint.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._lock = Lock()

        # Counters
        self.total_translation_api_calls: int = 0
        self.total_translation_cache_hits: int = 0
        self.total_tts_api_calls: int = 0
        self.total_segments_processed: int = 0
        self.total_jobs_completed: int = 0

        # Provider-level tracking
        self.provider_call_counts: dict[str, int] = {}

        # Timing
        self.started_at: float = time.time()

    def record_translation_call(self, provider: str, count: int = 1):
        """Record actual API translation calls made."""
        with self._lock:
            self.total_translation_api_calls += count
            self.provider_call_counts[provider] = (
                self.provider_call_counts.get(provider, 0) + count
            )

    def record_cache_hit(self, count: int = 1):
        """Record translation cache hits (API calls avoided)."""
        with self._lock:
            self.total_translation_cache_hits += count

    def record_tts_call(self, provider: str, count: int = 1):
        """Record TTS API calls made."""
        with self._lock:
            self.total_tts_api_calls += count
            key = f"{provider}_tts"
            self.provider_call_counts[key] = (
                self.provider_call_counts.get(key, 0) + count
            )

    def record_segments(self, count: int):
        """Record total segments processed."""
        with self._lock:
            self.total_segments_processed += count

    def record_job_completed(self):
        """Record a completed job."""
        with self._lock:
            self.total_jobs_completed += 1

    def get_snapshot(self) -> dict:
        """
        Get a full cost optimization snapshot for the /stats endpoint.

        Returns estimated costs, savings, and operational metrics.
        """
        with self._lock:
            calls_avoided = self.total_translation_cache_hits
            total_possible = self.total_translation_api_calls + calls_avoided
            hit_rate = (calls_avoided / total_possible * 100) if total_possible > 0 else 0.0

            # Estimate cost savings (using average provider cost)
            avg_cost_per_call = 0.003  # weighted average
            estimated_savings_usd = calls_avoided * avg_cost_per_call
            estimated_spent_usd = self.total_translation_api_calls * avg_cost_per_call

            # TTS cost estimation
            tts_costs = {}
            for key, count in self.provider_call_counts.items():
                if key.endswith("_tts"):
                    provider_name = key.replace("_tts", "")
                    rate = PROVIDER_COSTS.get(provider_name)
                    if rate:
                        tts_costs[provider_name] = {
                            "calls": count,
                            "estimated_cost_usd": round(count * rate.cost_per_tts_call, 4),
                        }

            uptime_seconds = time.time() - self.started_at

            return {
                "cost_optimization": {
                    "translation_api_calls_made": self.total_translation_api_calls,
                    "translation_api_calls_avoided_via_cache": calls_avoided,
                    "cache_hit_rate_pct": round(hit_rate, 1),
                    "estimated_translation_savings_usd": round(estimated_savings_usd, 4),
                    "estimated_translation_spent_usd": round(estimated_spent_usd, 4),
                },
                "tts_costs": tts_costs,
                "operations": {
                    "total_jobs_completed": self.total_jobs_completed,
                    "total_segments_processed": self.total_segments_processed,
                    "uptime_seconds": round(uptime_seconds, 0),
                },
                "provider_call_breakdown": dict(self.provider_call_counts),
            }

    def reset(self):
        """Reset all counters (for testing)."""
        with self._lock:
            self.total_translation_api_calls = 0
            self.total_translation_cache_hits = 0
            self.total_tts_api_calls = 0
            self.total_segments_processed = 0
            self.total_jobs_completed = 0
            self.provider_call_counts = {}
            self.started_at = time.time()


# Module-level singleton
cost_tracker = CostTracker()
