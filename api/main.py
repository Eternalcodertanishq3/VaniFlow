"""
FastAPI application with proper lifespan management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from vaaniflow.utils.logging import setup_logging
from vaaniflow.config import settings
from api.routes import jobs, health
from api.routes import metrics
from api.middleware.logging_middleware import LoggingMiddleware

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    setup_logging(settings.log_level)
    log.info("vaaniflow_starting", version="1.0.0", env=settings.environment)
    yield
    log.info("vaaniflow_shutting_down")


app = FastAPI(
    title="VaaniFlow",
    description="Multilingual async dubbing pipeline API — supports 11 Indian languages",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(metrics.router, tags=["observability"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        # We don't want to turn 404s/400s into 500s. We raise it to let FastAPI handle it properly
        # if it's already an HTTPException.
        raise

    log.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
