"""
IndianNamePronunciationCorrector — correct TTS pronunciation of Indian names/places.

Applied BEFORE sending text to TTS providers.
Corrects mispronunciations by substituting phonetic hints.
"""
import re
import structlog
from vaaniflow.pronunciation.indian_lexicon import INDIAN_PRONUNCIATION_MAP

log = structlog.get_logger(__name__)


class IndianNamePronunciationCorrector:
    """
    Pre-process text to fix TTS pronunciation of Indian words.
    Applied as a transformation step before TTS synthesis.
    """

    def __init__(self, enabled: bool = True, custom_map: dict = None):
        self.enabled = enabled
        self.pronunciation_map = {**INDIAN_PRONUNCIATION_MAP, **(custom_map or {})}
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple]:
        """Compile case-insensitive regex patterns for each entry."""
        patterns = []
        for original, phonetic in sorted(
            self.pronunciation_map.items(), key=lambda x: -len(x[0])
        ):
            pattern = re.compile(
                r'\b' + re.escape(original) + r'\b',
                re.IGNORECASE
            )
            patterns.append((pattern, phonetic, original))
        return patterns

    def correct(self, text: str) -> tuple[str, list[str]]:
        """
        Apply pronunciation corrections to text.

        Returns:
            (corrected_text, list_of_corrections_made)
        """
        if not self.enabled:
            return text, []

        corrections_made = []
        corrected = text

        for pattern, phonetic, original in self._patterns:
            if pattern.search(corrected):
                corrected = pattern.sub(phonetic, corrected)
                corrections_made.append(f"{original} -> {phonetic}")

        if corrections_made:
            log.debug(
                "pronunciation_corrections_applied",
                count=len(corrections_made),
                corrections=corrections_made[:3],
            )

        return corrected, corrections_made

    def add_correction(self, original: str, phonetic: str):
        """Add a custom pronunciation correction at runtime."""
        self.pronunciation_map[original] = phonetic
        self._patterns = self._compile_patterns()
        log.info("pronunciation_added", original=original, phonetic=phonetic)

    def remove_correction(self, original: str):
        """Remove a pronunciation correction."""
        if original in self.pronunciation_map:
            del self.pronunciation_map[original]
            self._patterns = self._compile_patterns()
