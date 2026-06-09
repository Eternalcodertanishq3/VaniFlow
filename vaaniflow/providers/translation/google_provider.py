"""
Google Translate API provider.
Uses the REST API via aiohttp with proper error handling.
"""
import asyncio
import aiohttp
import structlog

from vaaniflow.providers.translation.base import BaseTranslationProvider
from vaaniflow.models import SupportedLanguage
from vaaniflow.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderServerError,
    ProviderTimeoutError,
    TranslationError,
)
from vaaniflow.utils.retry import retry_on_rate_limit, retry_on_server_error, no_retry_on_auth_error
from vaaniflow.config import settings

log = structlog.get_logger(__name__)

GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_SUPPORTED_LANGUAGES = {
    "en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or",
}


class GoogleTranslationProvider(BaseTranslationProvider):
    """Google Cloud Translation API v2 provider."""

    provider_name = "google"

    def __init__(self):
        self.api_key = settings.google_translate_api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=settings.provider_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def supports_language(self, language_code: str) -> bool:
        return language_code in GOOGLE_SUPPORTED_LANGUAGES

    @no_retry_on_auth_error
    @retry_on_rate_limit(max_attempts=3)
    @retry_on_server_error(max_attempts=2)
    async def translate(
        self,
        text: str,
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> str:
        """Translate text using Google Translate API."""
        source = source_language.value if isinstance(source_language, SupportedLanguage) else source_language
        target = target_language.value if isinstance(target_language, SupportedLanguage) else target_language

        params = {
            "key": self.api_key,
            "q": text,
            "source": source,
            "target": target,
            "format": "text",
        }

        session = await self._get_session()
        try:
            async with session.post(GOOGLE_TRANSLATE_URL, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(self.provider_name, "Rate limited")
                if resp.status in (401, 403):
                    raise AuthenticationError(
                        self.provider_name, f"Invalid API key. Status: {resp.status}"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(
                        self.provider_name, f"Server error: {resp.status}"
                    )

                resp.raise_for_status()
                data = await resp.json()

                translations = data.get("data", {}).get("translations", [])
                if not translations:
                    raise TranslationError("No translation returned from Google API")

                translated_text = translations[0]["translatedText"]
                log.debug(
                    "google_translate_success",
                    source_lang=source,
                    target_lang=target,
                    input_length=len(text),
                    output_length=len(translated_text),
                )
                return translated_text

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                self.provider_name,
                f"Request timed out after {settings.provider_timeout_seconds}s",
            )

    async def health_check(self) -> bool:
        try:
            result = await self.translate("hello", "en", "hi")
            return bool(result)
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
