"""
Request/Response structured logging middleware.
Logs every HTTP request with method, path, status, and duration.
"""
import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every request/response with structured fields.
    Adds request_id tracking and duration measurement.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        # Extract request metadata
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        log.info(
            "http_request_started",
            method=method,
            path=path,
            client_ip=client_ip,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            log.info(
                "http_request_completed",
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add timing header to response
            response.headers["X-Process-Time-Ms"] = str(round(duration_ms, 2))
            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.error(
                "http_request_failed",
                method=method,
                path=path,
                error_type=type(e).__name__,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise
