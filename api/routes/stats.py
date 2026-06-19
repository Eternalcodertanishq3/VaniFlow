"""
Cost optimization & stats API route.

Exposes /stats endpoint showing:
  - Translation API calls made vs avoided (via Redis cache)
  - Estimated USD savings
  - TTS cost breakdown by provider
  - Operational metrics (jobs, segments, uptime)
"""
from fastapi import APIRouter
from vaaniflow.cost import cost_tracker

router = APIRouter()


@router.get("/stats")
async def get_cost_stats():
    """
    Token & Cost Optimization Dashboard.

    Returns real-time metrics on API call savings, cache efficiency,
    and estimated cost reduction — critical for enterprise deployments
    where cost-per-token at scale is a massive factor.
    """
    return cost_tracker.get_snapshot()
