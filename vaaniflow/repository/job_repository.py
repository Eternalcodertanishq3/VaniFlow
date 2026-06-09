"""
Redis-backed job repository.
Replaces in-memory dict — jobs survive server restarts.
"""
import json
from typing import Optional
import structlog
from vaaniflow.models import DubbingJob, JobStatus
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

JOB_KEY_PREFIX = "vaaniflow:job:"
JOB_TTL_SECONDS = 86400 * 7  # 7 days


class DubbingJobRepository:
    """
    Persists DubbingJob state in Redis.
    Falls back to in-memory store if Redis unavailable.
    """

    def __init__(self):
        self._redis = None
        self._fallback: dict[str, DubbingJob] = {}
        self._using_fallback = False

    async def _get_redis(self):
        if self._redis is None and not self._using_fallback:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    settings.redis_url, decode_responses=True, socket_connect_timeout=3
                )
                await self._redis.ping()
            except Exception as e:
                log.warning("job_repo_redis_unavailable", error=str(e))
                self._using_fallback = True
        return self._redis

    async def save(self, job: DubbingJob) -> bool:
        redis = await self._get_redis()
        if redis:
            try:
                key = f"{JOB_KEY_PREFIX}{job.job_id}"
                await redis.set(key, job.model_dump_json(), ex=JOB_TTL_SECONDS)
                return True
            except Exception as e:
                log.warning("job_save_failed", job_id=job.job_id, error=str(e))
        self._fallback[job.job_id] = job
        return True

    async def get(self, job_id: str) -> Optional[DubbingJob]:
        redis = await self._get_redis()
        if redis:
            try:
                raw = await redis.get(f"{JOB_KEY_PREFIX}{job_id}")
                if raw:
                    return DubbingJob.model_validate_json(raw)
            except Exception as e:
                log.warning("job_get_failed", job_id=job_id, error=str(e))
        return self._fallback.get(job_id)

    async def list_all(self) -> list[DubbingJob]:
        redis = await self._get_redis()
        if redis:
            try:
                keys = [k async for k in redis.scan_iter(f"{JOB_KEY_PREFIX}*")]
                if not keys:
                    return []
                values = await redis.mget(*keys)
                return [
                    DubbingJob.model_validate_json(v)
                    for v in values if v is not None
                ]
            except Exception as e:
                log.warning("job_list_failed", error=str(e))
        return list(self._fallback.values())

    async def delete(self, job_id: str) -> bool:
        redis = await self._get_redis()
        if redis:
            try:
                await redis.delete(f"{JOB_KEY_PREFIX}{job_id}")
            except Exception:
                pass
        self._fallback.pop(job_id, None)
        return True
