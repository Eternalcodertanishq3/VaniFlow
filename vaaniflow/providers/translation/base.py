"""
Abstract base class for translation providers.
"""
from abc import ABC, abstractmethod
import structlog

from vaaniflow.models import SupportedLanguage

log = structlog.get_logger(__name__)


class BaseTranslationProvider(ABC):
    """
    Abstract translation provider interface.
    All providers (Google, Sarvam) must implement these methods.
    """

    provider_name: str = "base"

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Text to translate.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            Translated text string.

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

    async def translate_batch(
        self,
        texts: list[str],
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> list[str]:
        """
        Translate multiple texts concurrently.
        Default implementation calls translate() N times concurrently (override for efficiency).
        """
        import asyncio
        tasks = [
            self.translate(text, source_language, target_language)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

    async def translate_with_logging(
        self,
        text: str,
        source_language: SupportedLanguage | str,
        target_language: SupportedLanguage | str,
    ) -> str:
        """Wrapper that adds structured logging to every translation call."""
        log.info(
            "translation_started",
            provider=self.provider_name,
            source_lang=str(source_language),
            target_lang=str(target_language),
            text_length=len(text),
        )
        try:
            result = await self.translate(text, source_language, target_language)
            log.info(
                "translation_completed",
                provider=self.provider_name,
                source_lang=str(source_language),
                target_lang=str(target_language),
                result_length=len(result),
            )
            return result
        except Exception as e:
            log.error(
                "translation_failed",
                provider=self.provider_name,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise
