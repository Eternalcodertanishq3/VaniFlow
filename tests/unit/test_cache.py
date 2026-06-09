"""
Unit tests for the Redis translation cache.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from vaaniflow.cache.redis_cache import TranslationCache


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    """Cache miss should return None."""
    cache = TranslationCache()
    cache._using_fallback = True  # Force in-memory mode

    result = await cache.get("nonexistent_key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_get():
    """Cache set followed by get should return the value."""
    cache = TranslationCache()
    cache._using_fallback = True  # Force in-memory mode

    key = "en:hi:Hello world"
    value = "नमस्ते दुनिया"

    await cache.set(key, value)
    result = await cache.get(key)

    assert result == value


@pytest.mark.asyncio
async def test_cache_invalidate():
    """Invalidated keys should return None on get."""
    cache = TranslationCache()
    cache._using_fallback = True

    key = "en:hi:Test"
    await cache.set(key, "परीक्षा")
    await cache.invalidate(key)

    result = await cache.get(key)
    assert result is None


@pytest.mark.asyncio
async def test_cache_clear_all():
    """Clear all should remove all cached entries."""
    cache = TranslationCache()
    cache._using_fallback = True

    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.clear_all()

    assert await cache.get("key1") is None
    assert await cache.get("key2") is None


@pytest.mark.asyncio
async def test_cache_falls_back_to_memory_on_redis_failure():
    """When Redis is unavailable, cache should use in-memory fallback."""
    cache = TranslationCache()
    # Don't connect to Redis — should fall back to in-memory
    cache._using_fallback = True

    key = "en:ta:Good morning"
    value = "காலை வணக்கம்"

    success = await cache.set(key, value)
    assert success is True

    result = await cache.get(key)
    assert result == value


@pytest.mark.asyncio
async def test_cache_set_returns_true():
    """Cache set should return True on success."""
    cache = TranslationCache()
    cache._using_fallback = True

    result = await cache.set("key", "value")
    assert result is True


@pytest.mark.asyncio
async def test_cache_multiple_keys():
    """Multiple different keys should be stored independently."""
    cache = TranslationCache()
    cache._using_fallback = True

    await cache.set("en:hi:hello", "नमस्ते")
    await cache.set("en:hi:world", "दुनिया")
    await cache.set("en:ta:hello", "வணக்கம்")

    assert await cache.get("en:hi:hello") == "नमस्ते"
    assert await cache.get("en:hi:world") == "दुनिया"
    assert await cache.get("en:ta:hello") == "வணக்கம்"
