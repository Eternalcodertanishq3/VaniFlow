"""
Retry logic using tenacity.
THIS IS CRITICAL — Sarvam JD specifically says:
"Improve error handling: distinguish rate limits from auth errors
from server failures, add fallback logic"

Each retry decorator handles a different failure type differently.
"""
import asyncio
from functools import wraps
from typing import Callable, Type
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    before_sleep_log,
    RetryError,
    after_log,
)
import logging

from vaaniflow.exceptions import (
    RateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    AuthenticationError,
)

log = structlog.get_logger(__name__)
_tenacity_logger = logging.getLogger("tenacity")


def retry_on_rate_limit(max_attempts: int = 3):
    """
    Retry on 429 Rate Limit errors with exponential backoff.
    Wait 4s, 8s, 16s between attempts.
    """
    return retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=16),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )


def retry_on_server_error(max_attempts: int = 3):
    """
    Retry on 5xx server errors with fixed 2s wait.
    Provider might be temporarily overloaded.
    """
    return retry(
        retry=retry_if_exception_type(ProviderServerError),
        wait=wait_fixed(2),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )


def retry_on_timeout(max_attempts: int = 2):
    """
    Retry on timeouts only once — if it times out twice, fall back.
    """
    return retry(
        retry=retry_if_exception_type(ProviderTimeoutError),
        wait=wait_fixed(1),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )


def no_retry_on_auth_error(func):
    """
    Auth errors (401/403) should NEVER be retried.
    Fail immediately and raise to the pipeline.
    This is a configuration issue, not a transient error.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AuthenticationError as e:
            log.error(
                "auth_error_no_retry",
                provider=e.provider,
                error=str(e),
                action="fail_immediately"
            )
            raise  # Do not retry, propagate up immediately
    return wrapper


def with_provider_fallback(primary_func, fallback_func):
    """
    Decorator factory for provider fallback.
    If primary fails after all retries, silently switch to fallback.
    """
    @wraps(primary_func)
    async def wrapper(*args, **kwargs):
        try:
            return await primary_func(*args, **kwargs)
        except (RetryError, Exception) as e:
            log.warning(
                "provider_fallback_triggered",
                primary=primary_func.__name__,
                fallback=fallback_func.__name__,
                original_error=str(e),
            )
            return await fallback_func(*args, **kwargs)
    return wrapper
