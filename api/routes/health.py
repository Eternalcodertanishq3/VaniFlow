"""
Health check endpoints — liveness, readiness, and dependency diagnostics.
"""
import asyncio
import shutil
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
        "version": "2.0.0",
        "environment": settings.environment,
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — checks all dependencies.
    Returns 200 always, but clearly marks what's available and what's missing.
    """
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    redis_ok = await _check_redis()

    checks = {
        "ffmpeg": {
            "available": ffmpeg_ok,
            "required": False,
            "note": (
                "Installed and ready"
                if ffmpeg_ok
                else "Not found - WAV-only mode. Install: winget install ffmpeg"
            ),
        },
        "redis": {
            "available": redis_ok,
            "required": False,
            "note": (
                "Connected"
                if redis_ok
                else "Not running - using in-memory fallback (jobs lost on restart)"
            ),
        },
        "whisper": {
            "available": await _check_whisper(),
            "required": True,
            "note": "Local transcription model",
        },
    }

    # System is "ready" if core components work (whisper)
    # ffmpeg and redis are optional (graceful degradation)
    all_critical_ready = checks["whisper"]["available"]

    supported_formats = [".wav"]
    if ffmpeg_ok:
        supported_formats.extend([".mp3", ".mp4", ".ogg", ".flac", ".m4a", ".webm", ".mkv"])
    else:
        # pydub might work for some formats even without ffmpeg
        supported_formats.extend([".mp3 (needs ffmpeg)", ".mp4 (needs ffmpeg)"])

    return {
        "status": "ready" if all_critical_ready else "degraded",
        "checks": checks,
        "supported_formats": supported_formats,
        "tips": _get_setup_tips(ffmpeg_ok, redis_ok),
    }


def _get_setup_tips(has_ffmpeg: bool, has_redis: bool) -> list[str]:
    """Generate actionable setup tips based on missing dependencies."""
    tips = []
    if not has_ffmpeg:
        tips.append(
            "Install ffmpeg for full audio/video support: "
            "winget install ffmpeg  OR  choco install ffmpeg  OR  "
            "download from https://ffmpeg.org/download.html"
        )
    if not has_redis:
        tips.append(
            "Start Redis for persistent job storage: "
            "docker run -d -p 6379:6379 redis:alpine  OR  "
            "install Redis locally. Without Redis, jobs are stored in-memory only."
        )
    if not tips:
        tips.append("All dependencies are properly configured!")
    return tips


async def _check_redis() -> bool:
    """Check Redis connectivity with a fast timeout."""
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=1)
        await r.ping()
        await r.close()
        return True
    except Exception:
        return False


async def _check_whisper() -> bool:
    """Check if faster-whisper is importable."""
    try:
        import faster_whisper
        return True
    except ImportError:
        return False
