"""
FastAPI application with proper lifespan management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
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
