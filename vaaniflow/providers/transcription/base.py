"""
Abstract base class for transcription providers.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import structlog

from vaaniflow.models import TranscriptionResult

log = structlog.get_logger(__name__)


class BaseTranscriptionProvider(ABC):
    """
    Abstract transcription provider interface.
    All providers (Whisper, AssemblyAI) must implement these methods.
    """

    provider_name: str = "base"

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, source_language: str
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text with segment timestamps.

        Args:
            audio_path: Path to audio file (WAV/MP3).
            source_language: ISO 639-1 language code.

        Returns:
            TranscriptionResult with timed segments.

        Raises:
            ProviderError subclasses on failure.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if provider is reachable."""
        ...

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Return True if this provider supports the given language."""
        ...

    async def transcribe_with_logging(
        self, audio_path: Path, source_language: str
    ) -> TranscriptionResult:
        """Wrapper that adds structured logging to every transcription call."""
        log.info(
            "transcription_started",
            provider=self.provider_name,
            language=source_language,
            audio_path=str(audio_path),
        )
        try:
            result = await self.transcribe(audio_path, source_language)
            log.info(
                "transcription_completed",
                provider=self.provider_name,
                segments=len(result.segments),
                total_duration_ms=result.total_duration_ms,
            )
            return result
        except Exception as e:
            log.error(
                "transcription_failed",
                provider=self.provider_name,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise
