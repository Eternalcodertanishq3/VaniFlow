"""
Async Redis cache for translation and TTS results.
Reduces API costs by caching repeated translations.
"""
import json
from typing import Optional
import structlog

from vaaniflow.config import settings

log = structlog.get_logger(__name__)


class TranslationCache:
    """
    Async Redis-backed cache for translation results.
    Falls back to in-memory dict if Redis is unavailable.
    """

    def __init__(self):
        self._redis = None
        self._fallback_cache: dict[str, str] = {}
        self._using_fallback = False

    async def _get_redis(self):
        """Lazy-initialize Redis connection."""
        if self._redis is None and not self._using_fallback:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                await self._redis.ping()
                log.info("redis_connected", url=settings.redis_url)
            except Exception as e:
                log.warning(
                    "redis_unavailable_using_fallback",
                    error=str(e),
                    fallback="in_memory_dict",
                )
                self._redis = None
                self._using_fallback = True
        return self._redis

    async def get(self, key: str) -> Optional[str]:
        """
        Get a cached translation result.

        Args:
            key: Cache key (format: "source_lang:target_lang:text")

        Returns:
            Cached translation string or None on miss.
        """
        redis = await self._get_redis()

        if redis:
            try:
                result = await redis.get(f"vaaniflow:translation:{key}")
                if result:
                    log.debug("cache_hit", key=key[:50])
                return result
            except Exception as e:
                log.warning("cache_get_error", error=str(e))
                return None
        else:
            # In-memory fallback
            return self._fallback_cache.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Cache a translation result.

        Args:
            key: Cache key (format: "source_lang:target_lang:text")
            value: Translation result to cache.
            ttl: Time-to-live in seconds. Defaults to config value.

        Returns:
            True if successfully cached.
        """
        ttl = ttl or settings.cache_ttl_seconds
        redis = await self._get_redis()

        if redis:
            try:
                await redis.set(
                    f"vaaniflow:translation:{key}",
                    value,
                    ex=ttl,
                )
                log.debug("cache_set", key=key[:50], ttl=ttl)
                return True
            except Exception as e:
                log.warning("cache_set_error", error=str(e))
                return False
        else:
            # In-memory fallback (no TTL support)
            self._fallback_cache[key] = value
            return True

    async def invalidate(self, key: str) -> bool:
        """Delete a cached entry."""
        redis = await self._get_redis()

        if redis:
            try:
                await redis.delete(f"vaaniflow:translation:{key}")
                return True
            except Exception as e:
                log.warning("cache_invalidate_error", error=str(e))
                return False
        else:
            self._fallback_cache.pop(key, None)
            return True

    async def clear_all(self) -> bool:
        """Clear all VaaniFlow cache entries."""
        redis = await self._get_redis()

        if redis:
            try:
                keys = []
                async for key in redis.scan_iter(match="vaaniflow:*"):
                    keys.append(key)
                if keys:
                    await redis.delete(*keys)
                log.info("cache_cleared", keys_removed=len(keys))
                return True
            except Exception as e:
                log.warning("cache_clear_error", error=str(e))
                return False
        else:
            self._fallback_cache.clear()
            return True

    async def close(self):
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            log.info("redis_connection_closed")
