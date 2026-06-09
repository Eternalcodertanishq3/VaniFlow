"""
Unit tests for retry logic.
Tests that different errors trigger the right retry behavior.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tenacity import RetryError

from vaaniflow.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderServerError,
    ProviderTimeoutError,
)
from vaaniflow.providers.tts.base import TTSSynthesisRequest, TTSSynthesisResponse


@pytest.mark.asyncio
async def test_auth_error_does_not_retry():
    """AuthenticationError must NOT trigger retry — fail immediately."""
    from vaaniflow.utils.retry import no_retry_on_auth_error

    call_count = 0

    @no_retry_on_auth_error
    async def always_auth_fail():
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("test_provider", "Invalid API key")

    with pytest.raises(AuthenticationError):
        await always_auth_fail()

    # Should only be called once — no retry on auth errors
    assert call_count == 1


@pytest.mark.asyncio
async def test_rate_limit_triggers_retry():
    """RateLimitError should trigger retry with exponential backoff."""
    from vaaniflow.utils.retry import retry_on_rate_limit

    call_count = 0

    @retry_on_rate_limit(max_attempts=3)
    async def rate_limited_func():
        nonlocal call_count
        call_count += 1
        raise RateLimitError("test_provider", "Rate limited")

    with pytest.raises(RateLimitError):
        await rate_limited_func()

    # Should be called max_attempts times
    assert call_count == 3


@pytest.mark.asyncio
async def test_server_error_retries_with_fixed_wait():
    """ProviderServerError should retry up to max_attempts."""
    from vaaniflow.utils.retry import retry_on_server_error

    call_count = 0

    @retry_on_server_error(max_attempts=2)
    async def server_error_func():
        nonlocal call_count
        call_count += 1
        raise ProviderServerError("test_provider", "Server down")

    with pytest.raises(ProviderServerError):
        await server_error_func()

    assert call_count == 2


@pytest.mark.asyncio
async def test_timeout_retries_once():
    """ProviderTimeoutError should retry only max_attempts times."""
    from vaaniflow.utils.retry import retry_on_timeout

    call_count = 0

    @retry_on_timeout(max_attempts=2)
    async def timeout_func():
        nonlocal call_count
        call_count += 1
        raise ProviderTimeoutError("test_provider", "Timed out")

    with pytest.raises(ProviderTimeoutError):
        await timeout_func()

    assert call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_success_after_retries():
    """RateLimitError should succeed if attempt < max_attempts passes."""
    from vaaniflow.utils.retry import retry_on_rate_limit

    call_count = 0

    @retry_on_rate_limit(max_attempts=3)
    async def eventually_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError("test_provider", "Rate limited")
        return "success"

    result = await eventually_succeeds()
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_provider_fallback():
    """with_provider_fallback should switch to fallback on primary failure."""
    from vaaniflow.utils.retry import with_provider_fallback

    async def primary_func():
        raise Exception("Primary failed")

    async def fallback_func():
        return "fallback_result"

    wrapped = with_provider_fallback(primary_func, fallback_func)
    result = await wrapped()
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_provider_fallback_uses_primary_on_success():
    """with_provider_fallback should use primary when it succeeds."""
    from vaaniflow.utils.retry import with_provider_fallback

    async def primary_func():
        return "primary_result"

    async def fallback_func():
        return "fallback_result"

    wrapped = with_provider_fallback(primary_func, fallback_func)
    result = await wrapped()
    assert result == "primary_result"
