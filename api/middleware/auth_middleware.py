"""
API key authentication middleware.
If VAANIFLOW_API_KEY is set, all non-health endpoints require X-API-Key header.
If not set (empty), auth is disabled for development convenience.
"""
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from vaaniflow.config import settings

log = structlog.get_logger(__name__)

# Paths that bypass authentication
AUTH_EXEMPT_PREFIXES = ("/health", "/metrics")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no API key is configured (dev mode)
        if not settings.api_key:
            return await call_next(request)

        # Skip auth for health and metrics endpoints
        path = request.url.path
        if any(path.startswith(prefix) for prefix in AUTH_EXEMPT_PREFIXES):
            return await call_next(request)

        # Validate API key
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != settings.api_key:
            log.warning(
                "auth_failed",
                path=path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
