"""
Custom exception hierarchy for VaaniFlow.
Sarvam specifically wants engineers who distinguish between
different failure types — rate limits vs auth errors vs server failures.
Every external API call must raise the right exception type.
"""


class VaaniFlowError(Exception):
    """Base exception for all VaaniFlow errors."""
    pass


# Provider errors — these are what retry logic acts on
class ProviderError(VaaniFlowError):
    """Base class for all provider failures."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    """HTTP 429 — provider is throttling us. Retry with backoff."""
    pass


class AuthenticationError(ProviderError):
    """HTTP 401/403 — bad API key. Do NOT retry, fail immediately."""
    pass


class ProviderServerError(ProviderError):
    """HTTP 500/502/503 — provider is down. Retry with backoff."""
    pass


class ProviderTimeoutError(ProviderError):
    """Request timed out. Retry once, then fall back to alternate provider."""
    pass


class ProviderUnavailableError(ProviderError):
    """Provider is completely unavailable. Use fallback immediately."""
    pass


# Pipeline errors
class PipelineError(VaaniFlowError):
    """Pipeline orchestration failed."""
    pass


class TranscriptionError(PipelineError):
    pass


class TranslationError(PipelineError):
    pass


class TTSError(PipelineError):
    pass


class AudioProcessingError(PipelineError):
    pass


# Config errors
class ConfigurationError(VaaniFlowError):
    """Invalid configuration provided."""
    pass
