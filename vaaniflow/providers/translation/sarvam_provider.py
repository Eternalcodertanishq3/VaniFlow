"""
Sarvam AI Translation API provider.
Primary provider for Indian language translation.
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

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_SUPPORTED_LANGUAGES = {
    "en", "hi", "bn", "te", "mr", "ta", "gu", "kn", "ml", "pa", "or",
}

# Sarvam uses its own language codes
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


class SarvamTranslationProvider(BaseTranslationProvider):
    """Sarvam AI Translation API provider — optimized for Indian languages."""

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
    async def translate(
        self,
        text: str,
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> str:
        """Translate text using Sarvam AI API."""
        source = source_language.value if isinstance(source_language, SupportedLanguage) else source_language
        target = target_language.value if isinstance(target_language, SupportedLanguage) else target_language

        source_sarvam = SARVAM_LANG_MAP.get(source, source)
        target_sarvam = SARVAM_LANG_MAP.get(target, target)

        payload = {
            "input": text,
            "source_language_code": source_sarvam,
            "target_language_code": target_sarvam,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True,
        }

        session = await self._get_session()
        try:
            async with session.post(SARVAM_TRANSLATE_URL, json=payload) as resp:
                if resp.status == 429:
                    raise RateLimitError(self.provider_name, "Rate limited by Sarvam API")
                if resp.status in (401, 403):
                    raise AuthenticationError(
                        self.provider_name, f"Invalid Sarvam API key. Status: {resp.status}"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(
                        self.provider_name, f"Sarvam server error: {resp.status}"
                    )

                resp.raise_for_status()
                data = await resp.json()

                translated_text = data.get("translated_text", "")
                if not translated_text:
                    raise TranslationError("No translation returned from Sarvam API")

                log.debug(
                    "sarvam_translate_success",
                    source_lang=source,
                    target_lang=target,
                    input_length=len(text),
                    output_length=len(translated_text),
                )
                return translated_text

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                self.provider_name,
                f"Sarvam request timed out after {settings.provider_timeout_seconds}s",
            )

    async def translate_batch(
        self,
        texts: list[str],
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> list[str]:
        """
        Sarvam API is single-text only.
        Default implementation: translate each text individually.
        """
        results = []
        for text in texts:
            result = await self.translate(text, source_language, target_language)
            results.append(result)
        return results

    async def health_check(self) -> bool:
        try:
            result = await self.translate("hello", "en", "hi")
            return bool(result)
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
