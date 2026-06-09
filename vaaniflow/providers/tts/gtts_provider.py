"""
gTTS (Google Text-to-Speech) free fallback provider.
Always available — no API key required.
Used as automatic fallback when primary TTS providers fail.
"""
import asyncio
import io
import structlog

from vaaniflow.providers.tts.base import BaseTTSProvider, TTSSynthesisRequest, TTSSynthesisResponse
from vaaniflow.exceptions import TTSError

log = structlog.get_logger(__name__)

GTTS_SUPPORTED_LANGUAGES = {
    "en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or",
}


class GTTSProvider(BaseTTSProvider):
    """
    gTTS free fallback TTS provider.
    No API key needed — always available as last resort.
    Quality is lower than ElevenLabs/Sarvam but guaranteed to work.
    """

    provider_name = "gtts"

    def supports_language(self, language_code: str) -> bool:
        return language_code in GTTS_SUPPORTED_LANGUAGES

    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        """
        Synthesize speech using gTTS.
        Runs in executor to avoid blocking the event loop (gTTS is synchronous).
        """
        try:
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._synthesize_sync, request.text, request.language
            )

            return TTSSynthesisResponse(
                audio_bytes=audio_bytes,
                duration_ms=len(audio_bytes) / 32.0,  # rough estimate
                provider=self.provider_name,
            )

        except Exception as e:
            raise TTSError(f"gTTS synthesis failed: {e}")

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        """Synchronous gTTS synthesis — runs in thread pool."""
        from gtts import gTTS

        tts = gTTS(text=text, lang=language, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()

        log.debug(
            "gtts_synthesis_done",
            language=language,
            text_length=len(text),
            audio_bytes=len(audio_bytes),
        )

        return audio_bytes

    async def health_check(self) -> bool:
        """gTTS is always available (uses Google Translate TTS endpoint)."""
        try:
            request = TTSSynthesisRequest(text="test", language="en")
            await self.synthesize(request)
            return True
        except Exception:
            return False
