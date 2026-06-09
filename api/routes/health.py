"""
Health check endpoints — liveness and readiness.
"""
from fastapi import APIRouter
import structlog

from vaaniflow.config import settings

router = APIRouter()
log = structlog.get_logger(__name__)


@router.get("/")
async def health_check():
    """
    Basic liveness probe.
    Returns 200 if the service is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.environment,
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — checks all dependencies.
    Returns 200 only if all critical services are reachable.
    """
    checks = {
        "redis": await _check_redis(),
        "ffmpeg": await _check_ffmpeg(),
    }

    all_ready = all(checks.values())

    if not all_ready:
        log.warning("readiness_check_failed", checks=checks)

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


async def _check_redis() -> bool:
    """Check Redis connectivity."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.close()
        return True
    except Exception:
        return False


async def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    import asyncio
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        return process.returncode == 0
    except Exception:
        return False
