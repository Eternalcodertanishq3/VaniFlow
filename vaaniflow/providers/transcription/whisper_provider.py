"""
Whisper transcription provider using faster-whisper.
Local inference — no API key needed.
"""
import asyncio
from pathlib import Path
import structlog

from vaaniflow.providers.transcription.base import BaseTranscriptionProvider
from vaaniflow.models import TranscriptionResult, AudioSegment, TranscriptionProvider
from vaaniflow.exceptions import TranscriptionError
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

WHISPER_SUPPORTED_LANGUAGES = {
    "en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or",
}


class WhisperProvider(BaseTranscriptionProvider):
    """
    faster-whisper local transcription provider.
    Runs inference locally — no API calls, no rate limits.
    """

    provider_name = "whisper"

    def __init__(self):
        self._model = None

    def _get_model(self):
        """Lazy-load the Whisper model to avoid loading at import time."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
                log.info(
                    "whisper_model_loaded",
                    model_size=settings.whisper_model_size,
                    device=settings.whisper_device,
                )
            except Exception as e:
                log.error("whisper_model_load_failed", error=str(e))
                raise TranscriptionError(f"Failed to load Whisper model: {e}")
        return self._model

    def supports_language(self, language_code: str) -> bool:
        return language_code in WHISPER_SUPPORTED_LANGUAGES

    async def transcribe(
        self, audio_path: Path, source_language: str
    ) -> TranscriptionResult:
        """
        Transcribe audio using faster-whisper.
        Runs in executor to avoid blocking the event loop.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_sync, audio_path, source_language
            )
            return result
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}")

    def _transcribe_sync(
        self, audio_path: Path, source_language: str
    ) -> TranscriptionResult:
        """Synchronous transcription — runs in thread pool."""
        model = self._get_model()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=source_language if source_language != "auto" else None,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        audio_segments = []
        for idx, segment in enumerate(segments_iter):
            start_ms = segment.start * 1000
            end_ms = segment.end * 1000
            audio_segments.append(
                AudioSegment(
                    index=idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    original_text=segment.text.strip(),
                )
            )

        total_duration_ms = audio_segments[-1].end_ms if audio_segments else 0.0

        log.info(
            "whisper_transcription_done",
            segments=len(audio_segments),
            detected_language=info.language,
            language_probability=round(info.language_probability, 3),
            total_duration_ms=total_duration_ms,
        )

        return TranscriptionResult(
            segments=audio_segments,
            source_language=info.language,
            total_duration_ms=total_duration_ms,
            provider_used=TranscriptionProvider.WHISPER,
        )

    async def health_check(self) -> bool:
        """Check if Whisper model can be loaded."""
        try:
            self._get_model()
            return True
        except Exception:
            return False
