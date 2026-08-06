"""
Google Translate API provider.
Uses official Google Cloud Translation REST API via aiohttp with proper error handling.
"""

import asyncio

import aiohttp
import structlog

from vaaniflow.config import settings
from vaaniflow.exceptions import (
    AuthenticationError,
    ProviderServerError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationError,
)
from vaaniflow.models import SupportedLanguage
from vaaniflow.providers.translation.base import BaseTranslationProvider
from vaaniflow.utils.retry import no_retry_on_auth_error, retry_on_rate_limit, retry_on_server_error

log = structlog.get_logger(__name__)

GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
GOOGLE_SUPPORTED_LANGUAGES = {
    "en",
    "hi",
    "bn",
    "te",
    "mr",
    "ta",
    "gu",
    "kn",
    "ml",
    "pa",
    "or",
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
        """Translate text using official Google Translate API."""
        source = (
            source_language.value
            if isinstance(source_language, SupportedLanguage)
            else source_language
        )
        target = (
            target_language.value
            if isinstance(target_language, SupportedLanguage)
            else target_language
        )

        if not self.api_key:
            log.info("google_translate_using_free_endpoint", text_length=len(text))
            return await self._translate_free_gtx(text, source, target)

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
                        self.provider_name, f"Invalid Google Translate API key (HTTP {resp.status})"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(self.provider_name, f"Server error: {resp.status}")

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

    @retry_on_rate_limit(max_attempts=3)
    @retry_on_server_error(max_attempts=2)
    async def translate_batch(
        self,
        texts: list[str],
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> list[str]:
        """
        Google Translate batch API — one call for all segments.
        MUCH cheaper and faster than N individual calls.
        """
        source = (
            source_language.value
            if isinstance(source_language, SupportedLanguage)
            else source_language
        )
        target = (
            target_language.value
            if isinstance(target_language, SupportedLanguage)
            else target_language
        )

        if not self.api_key:
            log.info("google_translate_batch_using_free_endpoint", count=len(texts))
            tasks = [self._translate_free_gtx(text, source, target) for text in texts]
            return await asyncio.gather(*tasks)

        params = {
            "key": self.api_key,
            "target": target,
            "source": source,
            "format": "text",
        }

        session = await self._get_session()
        try:
            params["q"] = texts
            async with session.post(GOOGLE_TRANSLATE_URL, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(self.provider_name, "Rate limited")
                if resp.status in (401, 403):
                    raise AuthenticationError(
                        self.provider_name, f"Invalid Google Translate API key (HTTP {resp.status})"
                    )
                if resp.status >= 500:
                    raise ProviderServerError(self.provider_name, f"Server error: {resp.status}")

                resp.raise_for_status()
                data = await resp.json()
                translations = data.get("data", {}).get("translations", [])
                return [t["translatedText"] for t in translations]

        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                self.provider_name,
                f"Batch request timed out after {settings.provider_timeout_seconds}s",
            )

    async def _translate_free_gtx(self, text: str, source: str, target: str) -> str:
        """Free Google Translate endpoint — zero API keys required."""
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
            "q": text,
        }
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                translated_text = "".join(item[0] for item in data[0] if item and item[0])
                return translated_text or text
        except Exception as e:
            log.warning("free_google_translate_failed", error=str(e))
            raise TranslationError(f"Free Google translation failed: {e}") from e

    async def health_check(self) -> bool:
        try:
            result = await self.translate("hello", "en", "hi")
            return bool(result)
        except Exception:
            return False

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
