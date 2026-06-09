"""
Sarvam AI TTS Provider implementation.
Primary provider for Indian language text-to-speech.
"""
import asyncio
import aiohttp
import structlog

from vaaniflow.providers.tts.base import BaseTTSProvider, TTSSynthesisRequest, TTSSynthesisResponse
from vaaniflow.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderServerError,
    ProviderTimeoutError,
    TTSError,
)
from vaaniflow.utils.retry import retry_on_rate_limit, retry_on_server_error, no_retry_on_auth_error
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_SUPPORTED_LANGUAGES = {
    "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or", "en",
}

# Default Sarvam voice per language
SARVAM_DEFAULT_VOICES = {
    "hi": "arvind",
    "bn": "arvind",
    "te": "arvind",
    "mr": "arvind",
    "ta": "arvind",
    "gu": "arvind",
    "kn": "arvind",
    "ml": "arvind",
    "pa": "arvind",
    "or": "arvind",
    "en": "arvind",
}

# Sarvam language code mapping
SARVAM_LANG_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "or": "or-IN",
}


class SarvamTTSProvider(BaseTTSProvider):
    """Sarvam AI TTS provider — optimized for Indian languages."""

    provider_name = "sarvam"

    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=settings.provider_timeout_seconds)
            self._session = aiohttp.ClientSession(
                headers={
                    "API-Subscription-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
        return self._session

    def supports_language(self, language_code: str) -> bool:
        return language_code in SARVAM_SUPPORTED_LANGUAGES

    @no_retry_on_auth_error
    @retry_on_rate_limit(max_attempts=3)
    @retry_on_server_error(max_attempts=2)
    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        """Synthesize speech using Sarvam AI TTS API."""
        lang_code = request.language
        sarvam_lang = SARVAM_LANG_MAP.get(lang_code, lang_code)
        voice = request.voice_id or SARVAM_DEFAULT_VOICES.get(lang_code, "arvind")

        payload = {
            "inputs": [request.text],
            "target_language_code": sarvam_lang,
            "speaker": voice,
            "pitch": request.pitch,
            "pace": request.speaking_rate,
            "loudness": 1.5,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1",
        }

        session = await self._get_session()
        try:
            async with session.post(SARVAM_TTS_URL, json=payload) as resp:
                if resp.status == 429:
                    raise RateLimitError(self.provider_name, "Rate limited by Sarvam TTS")
                if resp.status in (401, 403):
                    raise AuthenticationError(
                        self.provider_name, f"Invalid Sarvam API key. Status: {resp.status}"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(
                        self.provider_name, f"Sarvam TTS server error: {resp.status}"
                    )

                resp.raise_for_status()
                data = await resp.json()

                # Sarvam returns base64-encoded audio
                audios = data.get("audios", [])
                if not audios:
                    raise TTSError("No audio returned from Sarvam TTS")

                import base64
                audio_bytes = base64.b64decode(audios[0])

                return TTSSynthesisResponse(
                    audio_bytes=audio_bytes,
                    duration_ms=len(audio_bytes) / 44.1,  # rough estimate at 22050Hz 16-bit
                    provider=self.provider_name,
                )

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                self.provider_name,
                f"Sarvam TTS timed out after {settings.provider_timeout_seconds}s",
            )

    async def health_check(self) -> bool:
        """Check if Sarvam TTS is reachable."""
        try:
            request = TTSSynthesisRequest(text="test", language="hi")
            await self.synthesize(request)
            return True
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
