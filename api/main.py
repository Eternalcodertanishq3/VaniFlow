"""
FastAPI application with proper lifespan management.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api.middleware.auth_middleware import APIKeyAuthMiddleware
from api.middleware.logging_middleware import LoggingMiddleware
from api.routes import health, jobs, metrics, stats, voices
from vaaniflow.config import settings
from vaaniflow.utils.logging import setup_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    setup_logging(settings.log_level)
    log.info("vaaniflow_starting", version="2.0.0", env=settings.environment)
    yield
    log.info("vaaniflow_shutting_down")


app = FastAPI(
    title="VaaniFlow",
    description="Multilingual async dubbing pipeline API — supports 11 Indian languages",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyAuthMiddleware)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(voices.router, prefix="/voices", tags=["voices"])
app.include_router(metrics.router, tags=["observability"])
app.include_router(stats.router, tags=["cost-optimization"])

UI_INDEX_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serve single-page HTML web interface."""
    if UI_INDEX_PATH.exists():
        return FileResponse(str(UI_INDEX_PATH))
    return JSONResponse({"message": "VaaniFlow API v2.0.0. UI not found."})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    log.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
