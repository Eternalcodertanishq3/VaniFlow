"""
Abstract base class for TTS providers.
THIS IS THE KEY PATTERN Sarvam mentions: "provider abstractions"
Every provider must implement this interface identically.
The pipeline never depends on a concrete provider — only this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import structlog

log = structlog.get_logger(__name__)


@dataclass
class TTSSynthesisRequest:
    text: str
    language: str
    voice_id: str | None = None
    speaking_rate: float = 1.0
    pitch: float = 0.0


@dataclass
class TTSSynthesisResponse:
    audio_bytes: bytes
    duration_ms: float
    provider: str
    cached: bool = False


class BaseTTSProvider(ABC):
    """
    Abstract TTS provider interface.
    All providers (ElevenLabs, Sarvam, gTTS) must implement these methods.
    The pipeline only ever calls methods on this interface.
    """
    provider_name: str = "base"

    @abstractmethod
    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        """
        Convert text to speech audio bytes.
        Must raise:
          - RateLimitError on HTTP 429
          - AuthenticationError on HTTP 401/403
          - ProviderServerError on HTTP 5xx
          - ProviderTimeoutError on timeout
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if provider is reachable and authenticated."""
        ...

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Return True if this provider supports the given language."""
        ...

    async def synthesize_with_logging(
        self, request: TTSSynthesisRequest
    ) -> TTSSynthesisResponse:
        """Wrapper that adds structured logging to every synthesis call."""
        log.info(
            "tts_synthesis_started",
            provider=self.provider_name,
            language=request.language,
            text_length=len(request.text),
        )
        try:
            result = await self.synthesize(request)
            log.info(
                "tts_synthesis_completed",
                provider=self.provider_name,
                language=request.language,
                audio_bytes=len(result.audio_bytes),
                duration_ms=result.duration_ms,
                cached=result.cached,
            )
            return result
        except Exception as e:
            log.error(
                "tts_synthesis_failed",
                provider=self.provider_name,
                language=request.language,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise
