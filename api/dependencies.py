"""
FastAPI dependency injection.
Provides pipeline, cache, and settings as injectable dependencies.
"""
from functools import lru_cache
from vaaniflow.config import Settings, settings
from vaaniflow.pipeline import VaaniFlowPipeline
from vaaniflow.cache.redis_cache import TranslationCache


# Singleton pipeline instance
_pipeline: VaaniFlowPipeline | None = None
_cache: TranslationCache | None = None


def get_settings() -> Settings:
    """Dependency: inject application settings."""
    return settings


def get_pipeline() -> VaaniFlowPipeline:
    """Dependency: inject the dubbing pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VaaniFlowPipeline()
    return _pipeline


def get_cache() -> TranslationCache:
    """Dependency: inject the translation cache (singleton)."""
    global _cache
    if _cache is None:
        _cache = TranslationCache()
    return _cache
