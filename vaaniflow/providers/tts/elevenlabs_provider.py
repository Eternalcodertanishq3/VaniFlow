"""
ElevenLabs TTS Provider implementation.
Shows how to translate HTTP errors into our custom exception hierarchy.
"""
import asyncio
import aiohttp
import structlog
from vaaniflow.providers.tts.base import BaseTTSProvider, TTSSynthesisRequest, TTSSynthesisResponse
from vaaniflow.exceptions import RateLimitError, AuthenticationError, ProviderServerError, ProviderTimeoutError
from vaaniflow.utils.retry import retry_on_rate_limit, retry_on_server_error, no_retry_on_auth_error
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

ELEVENLABS_SUPPORTED_LANGUAGES = {"en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu"}

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


class ElevenLabsProvider(BaseTTSProvider):
    provider_name = "elevenlabs"

    def __init__(self):
        self.api_key = settings.elevenlabs_api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=settings.provider_timeout_seconds)
            self._session = aiohttp.ClientSession(
                headers={"xi-api-key": self.api_key},
                timeout=timeout,
            )
        return self._session

    def supports_language(self, language_code: str) -> bool:
        return language_code in ELEVENLABS_SUPPORTED_LANGUAGES

    @no_retry_on_auth_error
    @retry_on_rate_limit(max_attempts=3)
    @retry_on_server_error(max_attempts=2)
    async def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResponse:
        voice_id = request.voice_id or DEFAULT_VOICE_ID
        url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
        payload = {
            "text": request.text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speaking_rate": request.speaking_rate,
            },
        }

        session = await self._get_session()
        try:
            async with session.post(url, json=payload) as resp:
                # Distinguish error types — this is exactly what Sarvam wants
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "unknown")
                    raise RateLimitError(
                        self.provider_name,
                        f"Rate limited. Retry-After: {retry_after}"
                    )
                if resp.status in (401, 403):
                    raise AuthenticationError(
                        self.provider_name,
                        f"Invalid API key. Status: {resp.status}"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(
                        self.provider_name,
                        f"Server error. Status: {resp.status}"
                    )

                resp.raise_for_status()
                audio_bytes = await resp.read()

                return TTSSynthesisResponse(
                    audio_bytes=audio_bytes,
                    duration_ms=len(audio_bytes) / 32.0,  # rough estimate
                    provider=self.provider_name,
                )

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                self.provider_name,
                f"Request timed out after {settings.provider_timeout_seconds}s"
            )

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get("https://api.elevenlabs.io/v1/user") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
