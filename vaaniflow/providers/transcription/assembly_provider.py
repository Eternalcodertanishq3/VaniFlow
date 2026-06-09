"""
AssemblyAI cloud transcription provider.
Cloud fallback when local Whisper is unavailable or too slow.
"""
import asyncio
import aiohttp
import structlog

from pathlib import Path

from vaaniflow.providers.transcription.base import BaseTranscriptionProvider
from vaaniflow.models import TranscriptionResult, AudioSegment, TranscriptionProvider
from vaaniflow.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderServerError,
    ProviderTimeoutError,
    TranscriptionError,
)
from vaaniflow.utils.retry import retry_on_rate_limit, retry_on_server_error, no_retry_on_auth_error
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
ASSEMBLYAI_SUPPORTED_LANGUAGES = {
    "en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa",
}


class AssemblyAIProvider(BaseTranscriptionProvider):
    """
    AssemblyAI cloud transcription provider.
    Used as fallback when local Whisper isn't available.
    """

    provider_name = "assemblyai"

    def __init__(self):
        self.api_key = settings.assemblyai_api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)  # Transcription can be slow
            self._session = aiohttp.ClientSession(
                headers={"Authorization": self.api_key},
                timeout=timeout,
            )
        return self._session

    def supports_language(self, language_code: str) -> bool:
        return language_code in ASSEMBLYAI_SUPPORTED_LANGUAGES

    @no_retry_on_auth_error
    @retry_on_rate_limit(max_attempts=3)
    @retry_on_server_error(max_attempts=2)
    async def transcribe(
        self, audio_path: Path, source_language: str
    ) -> TranscriptionResult:
        """Upload audio and get transcription from AssemblyAI."""
        session = await self._get_session()

        try:
            # Step 1: Upload audio file
            upload_url = await self._upload_file(session, audio_path)
            log.info("assemblyai_file_uploaded", upload_url=upload_url[:50])

            # Step 2: Create transcription request
            payload = {
                "audio_url": upload_url,
                "language_code": source_language,
            }

            async with session.post(ASSEMBLYAI_TRANSCRIPT_URL, json=payload) as resp:
                self._handle_http_error(resp.status)
                data = await resp.json()
                transcript_id = data["id"]

            # Step 3: Poll for completion
            result = await self._poll_transcription(session, transcript_id)
            return result

        except (RateLimitError, AuthenticationError, ProviderServerError):
            raise
        except asyncio.TimeoutError:
            raise ProviderTimeoutError(self.provider_name, "Request timed out")
        except Exception as e:
            raise TranscriptionError(f"AssemblyAI transcription failed: {e}")

    async def _upload_file(
        self, session: aiohttp.ClientSession, audio_path: Path
    ) -> str:
        """Upload audio file to AssemblyAI and return the upload URL."""
        with open(audio_path, "rb") as f:
            async with session.post(ASSEMBLYAI_UPLOAD_URL, data=f) as resp:
                self._handle_http_error(resp.status)
                data = await resp.json()
                return data["upload_url"]

    async def _poll_transcription(
        self,
        session: aiohttp.ClientSession,
        transcript_id: str,
        max_polls: int = 60,
        poll_interval: float = 3.0,
    ) -> TranscriptionResult:
        """Poll AssemblyAI until transcription is complete."""
        url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"

        for _ in range(max_polls):
            async with session.get(url) as resp:
                self._handle_http_error(resp.status)
                data = await resp.json()

            status = data["status"]
            if status == "completed":
                return self._parse_result(data)
            elif status == "error":
                raise TranscriptionError(
                    f"AssemblyAI error: {data.get('error', 'Unknown')}"
                )

            await asyncio.sleep(poll_interval)

        raise ProviderTimeoutError(
            self.provider_name, "Transcription polling timed out"
        )

    def _parse_result(self, data: dict) -> TranscriptionResult:
        """Parse AssemblyAI response into our TranscriptionResult model."""
        segments = []
        for idx, utterance in enumerate(data.get("utterances", data.get("words", []))):
            start_ms = utterance.get("start", 0)
            end_ms = utterance.get("end", 0)
            segments.append(
                AudioSegment(
                    index=idx,
                    start_ms=float(start_ms),
                    end_ms=float(end_ms),
                    duration_ms=float(end_ms - start_ms),
                    original_text=utterance.get("text", ""),
                )
            )

        # If no utterances, create a single segment from full text
        if not segments and data.get("text"):
            audio_duration = data.get("audio_duration", 0) * 1000
            segments.append(
                AudioSegment(
                    index=0,
                    start_ms=0,
                    end_ms=audio_duration,
                    duration_ms=audio_duration,
                    original_text=data["text"],
                )
            )

        total_duration = data.get("audio_duration", 0) * 1000

        return TranscriptionResult(
            segments=segments,
            source_language=data.get("language_code", "en"),
            total_duration_ms=float(total_duration),
            provider_used=TranscriptionProvider.ASSEMBLY,
        )

    def _handle_http_error(self, status: int):
        """Map HTTP status codes to our exception hierarchy."""
        if status == 429:
            raise RateLimitError(self.provider_name, "Rate limited")
        if status in (401, 403):
            raise AuthenticationError(self.provider_name, f"Auth failed: {status}")
        if status >= 500:
            raise ProviderServerError(self.provider_name, f"Server error: {status}")

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(ASSEMBLYAI_TRANSCRIPT_URL) as resp:
                return resp.status in (200, 401)  # 401 means API is up, key may be bad
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
