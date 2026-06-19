"""
CodeSwitchNormalizer — detect and normalize code-switched (Hinglish/Tanglish) text.

Detects English (Latin-script) words embedded within Indic-script text and marks
them with [EN:word] tags so downstream TTS providers can apply language-appropriate
pronunciation.  Also normalizes common code-switched phrases to their canonical forms.

Applied BEFORE sending text to TTS providers.
"""
import re
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Unicode ranges for Indic scripts
# ---------------------------------------------------------------------------
_DEVANAGARI_RANGE = r"\u0900-\u097F"   # Hindi, Marathi, Sanskrit
_TAMIL_RANGE = r"\u0B80-\u0BFF"        # Tamil
_TELUGU_RANGE = r"\u0C00-\u0C7F"       # Telugu
_BENGALI_RANGE = r"\u0980-\u09FF"      # Bengali
_GUJARATI_RANGE = r"\u0A80-\u0AFF"     # Gujarati
_KANNADA_RANGE = r"\u0C80-\u0CFF"      # Kannada
_MALAYALAM_RANGE = r"\u0D00-\u0D7F"    # Malayalam
_GURMUKHI_RANGE = r"\u0A00-\u0A7F"     # Punjabi

_ALL_INDIC_RANGES = (
    _DEVANAGARI_RANGE + _TAMIL_RANGE + _TELUGU_RANGE + _BENGALI_RANGE
    + _GUJARATI_RANGE + _KANNADA_RANGE + _MALAYALAM_RANGE + _GURMUKHI_RANGE
)

# Regex: a token made entirely of Indic-script characters (+ common combining marks)
_INDIC_TOKEN_RE = re.compile(
    rf"[{_ALL_INDIC_RANGES}\u0900-\u0D7F]+", re.UNICODE
)

# Regex: a token made entirely of Basic-Latin letters (ASCII A-Za-z, optionally
# with apostrophe/hyphen for contractions like "don't" or "e-mail")
_LATIN_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]", re.UNICODE
)

# Regex: a purely numeric token (digits, commas, decimals)
_NUMERIC_TOKEN_RE = re.compile(r"^[\d,\.]+$")

# ---------------------------------------------------------------------------
# Common code-switched phrases  (source → normalised form)
# Keys are lowercase for case-insensitive matching.
# ---------------------------------------------------------------------------
COMMON_CODE_SWITCH_PHRASES: dict[str, str] = {
    "bill print karo":          "bill print करो",
    "please confirm kardo":     "please confirm कर दो",
    "please confirm kar do":    "please confirm कर दो",
    "meeting schedule karo":    "meeting schedule करो",
    "email send karo":          "email send करो",
    "file download karo":       "file download करो",
    "password reset karo":      "password reset करो",
    "order cancel karo":        "order cancel करो",
    "payment process karo":     "payment process करो",
    "update install karo":      "update install करो",
    "report generate karo":     "report generate करो",
    "data backup karo":         "data backup करो",
}


class CodeSwitchNormalizer:
    """
    Detect English tokens inside Indic-script text and mark them with
    ``[EN:word]`` tags so TTS providers can switch voice/language mid-utterance.

    Also applies common code-switched phrase normalisations (e.g. Romanised
    Hindi verbs → Devanagari).
    """

    def __init__(self, enabled: bool = True, custom_phrases: dict = None):
        self.enabled = enabled
        self.phrase_map = {**COMMON_CODE_SWITCH_PHRASES, **(custom_phrases or {})}
        self._phrase_patterns = self._compile_phrase_patterns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_phrase_patterns(self) -> list[tuple]:
        """Compile case-insensitive regex patterns for each known phrase.

        Longer phrases are compiled first so they match before shorter
        sub-phrases — identical strategy to the pronunciation corrector.
        """
        patterns = []
        for original, normalised in sorted(
            self.phrase_map.items(), key=lambda x: -len(x[0])
        ):
            pattern = re.compile(
                r'\b' + re.escape(original) + r'\b',
                re.IGNORECASE,
            )
            patterns.append((pattern, normalised, original))
        return patterns

    @staticmethod
    def _text_has_indic(text: str) -> bool:
        """Return True if *text* contains at least one Indic-script character."""
        return bool(_INDIC_TOKEN_RE.search(text))

    @staticmethod
    def _is_numeric(token: str) -> bool:
        """Return True if *token* is purely numeric (digits, commas, dots)."""
        return bool(_NUMERIC_TOKEN_RE.match(token))

    @staticmethod
    def _mark_english(token: str) -> str:
        """Wrap *token* in the ``[EN:…]`` marker for downstream TTS."""
        return f"[EN:{token}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self, text: str, target_language: str = "hi"
    ) -> tuple[str, list[str]]:
        """Normalise code-switched text for TTS.

        Args:
            text: Input text, possibly mixing Latin and Indic scripts.
            target_language: BCP-47 language code of the target TTS voice
                (e.g. ``"hi"`` for Hindi, ``"ta"`` for Tamil).

        Returns:
            A tuple of ``(normalised_text, list_of_normalisations_applied)``.
            When the normaliser is disabled the original text is returned
            unchanged with an empty list.
        """
        if not self.enabled:
            return text, []

        if not text or not text.strip():
            return text, []

        normalizations: list[str] = []
        result = text

        # --- Step 1: Apply known phrase normalisations -----------------
        for pattern, normalised, original in self._phrase_patterns:
            if pattern.search(result):
                result = pattern.sub(normalised, result)
                normalizations.append(f"phrase: {original} -> {normalised}")

        # --- Step 2: Mark remaining English tokens in Indic context ----
        if self._text_has_indic(result):
            result = self._mark_english_tokens(result, normalizations)

        if normalizations:
            log.debug(
                "code_switch_normalizations_applied",
                target_language=target_language,
                count=len(normalizations),
                normalizations=normalizations[:5],
            )

        return result, normalizations

    def _mark_english_tokens(
        self, text: str, normalizations: list[str]
    ) -> str:
        """Walk through *text* and wrap un-marked Latin-script tokens
        with ``[EN:…]`` when Indic context is present."""
        parts: list[str] = []
        last_end = 0

        for match in _LATIN_TOKEN_RE.finditer(text):
            token = match.group()
            start, end = match.start(), match.end()

            # Keep everything between the previous match end and this start
            parts.append(text[last_end:start])

            # Skip tokens that are already inside an [EN:…] marker
            prefix = text[max(0, start - 4):start]
            if prefix.endswith("[EN:"):
                parts.append(token)
                last_end = end
                continue

            # Skip numeric-looking tokens
            if self._is_numeric(token):
                parts.append(token)
                last_end = end
                continue

            # Mark the English token
            marked = self._mark_english(token)
            parts.append(marked)
            normalizations.append(f"en_mark: {token} -> {marked}")
            last_end = end

        # Append any trailing text
        parts.append(text[last_end:])
        return "".join(parts)

    # ------------------------------------------------------------------
    # Runtime mutation (mirrors IndianNamePronunciationCorrector)
    # ------------------------------------------------------------------

    def add_phrase(self, phrase: str, normalised: str):
        """Add a custom code-switch phrase at runtime."""
        self.phrase_map[phrase.lower()] = normalised
        self._phrase_patterns = self._compile_phrase_patterns()
        log.info("code_switch_phrase_added", phrase=phrase, normalised=normalised)

    def remove_phrase(self, phrase: str):
        """Remove a code-switch phrase."""
        key = phrase.lower()
        if key in self.phrase_map:
            del self.phrase_map[key]
            self._phrase_patterns = self._compile_phrase_patterns()
