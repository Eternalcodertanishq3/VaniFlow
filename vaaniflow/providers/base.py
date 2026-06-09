"""
Base provider interface.
All concrete providers (transcription, translation, TTS) inherit from this.
"""
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Root base class for all VaaniFlow providers."""

    provider_name: str = "base"

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if provider is reachable and authenticated."""
        ...

    @abstractmethod
    def supports_language(self, language_code: str) -> bool:
        """Return True if this provider supports the given language."""
        ...
