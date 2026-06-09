"""
Prometheus metrics endpoint.
Exposes /metrics for scraping by Prometheus/Grafana.
Serves the library-level metrics defined in vaaniflow/metrics.py.
"""
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint — scrape this with Prometheus."""
    # Import to ensure metrics are registered
    import vaaniflow.metrics  # noqa: F401
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
