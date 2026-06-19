"""
Unit tests for CodeSwitchNormalizer.
Tests Hinglish/Tanglish detection, English-token marking, phrase normalisation,
and edge cases (pure scripts, numbers, disabled mode).
"""
import pytest
from vaaniflow.normalization.code_switch_normalizer import (
    CodeSwitchNormalizer,
    COMMON_CODE_SWITCH_PHRASES,
)


@pytest.fixture
def normalizer():
    return CodeSwitchNormalizer(enabled=True)


@pytest.fixture
def disabled_normalizer():
    return CodeSwitchNormalizer(enabled=False)


# ------------------------------------------------------------------
# Basic toggle / passthrough
# ------------------------------------------------------------------


def test_disabled_returns_unchanged(disabled_normalizer):
    """When enabled=False, text is returned as-is with no normalizations."""
    text = "Bill print करो meeting schedule करो"
    result, norms = disabled_normalizer.normalize(text, target_language="hi")
    assert result == text
    assert norms == []


def test_empty_text(normalizer):
    """Empty / whitespace-only text passes through unchanged."""
    result, norms = normalizer.normalize("", target_language="hi")
    assert result == ""
    assert norms == []

    result2, norms2 = normalizer.normalize("   ", target_language="hi")
    assert result2 == "   "
    assert norms2 == []


# ------------------------------------------------------------------
# Pure-script inputs (no code-switching expected)
# ------------------------------------------------------------------


def test_pure_english_unchanged(normalizer):
    """Pure English text should not be modified (no Indic context)."""
    text = "The quick brown fox jumps over the lazy dog"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert result == text
    assert norms == []


def test_pure_hindi_unchanged(normalizer):
    """Pure Devanagari text should not be modified."""
    text = "यह एक परीक्षण वाक्य है"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert result == text
    assert norms == []


def test_pure_tamil_unchanged(normalizer):
    """Pure Tamil text should not be modified."""
    text = "இது ஒரு சோதனை வாக்கியம்"
    result, norms = normalizer.normalize(text, target_language="ta")
    assert result == text
    assert norms == []


# ------------------------------------------------------------------
# Hinglish detection & marking
# ------------------------------------------------------------------


def test_hinglish_detection(normalizer):
    """'Bill print karo' in a Hinglish context triggers phrase normalisation."""
    text = "Bill print karo"
    result, norms = normalizer.normalize(text, target_language="hi")
    # The phrase map converts "bill print karo" -> "bill print करो"
    assert "करो" in result
    assert len(norms) >= 1


def test_mixed_script_marking(normalizer):
    """English words in Devanagari context get marked with [EN:...]."""
    text = "मुझे email भेजो"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "[EN:email]" in result
    assert any("en_mark" in n for n in norms)


def test_multiple_english_words_in_hindi(normalizer):
    """Multiple English insertions inside Hindi should all be marked."""
    text = "Meeting schedule करो please"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "[EN:Meeting]" in result
    assert "[EN:schedule]" in result
    assert "[EN:please]" in result
    # Verify each marking generated a normalisation entry
    en_marks = [n for n in norms if n.startswith("en_mark")]
    assert len(en_marks) == 3


def test_single_english_word_in_tamil(normalizer):
    """A single English word in Tamil context should be marked."""
    text = "இந்த file பாருங்கள்"
    result, norms = normalizer.normalize(text, target_language="ta")
    assert "[EN:file]" in result


# ------------------------------------------------------------------
# Common phrase normalisation
# ------------------------------------------------------------------


def test_common_phrases_normalized(normalizer):
    """Phrases from COMMON_CODE_SWITCH_PHRASES are normalised."""
    text = "Please confirm kardo"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "कर दो" in result
    assert any("phrase:" in n for n in norms)


def test_phrase_case_insensitive(normalizer):
    """Phrase matching should be case-insensitive."""
    text = "EMAIL SEND KARO"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "करो" in result


def test_common_phrases_dict_has_entries():
    """Phrase dictionary should have a reasonable number of entries."""
    assert len(COMMON_CODE_SWITCH_PHRASES) >= 10


# ------------------------------------------------------------------
# Numbers / edge cases
# ------------------------------------------------------------------


def test_numbers_not_marked(normalizer):
    """Numeric tokens like '500' should NOT be marked as English."""
    text = "मुझे 500 रुपये दो"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "[EN:500]" not in result
    assert "500" in result


def test_number_with_commas_not_marked(normalizer):
    """Formatted numbers like '1,000' should not be marked."""
    text = "कीमत 1,000 है"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "[EN:" not in result


# ------------------------------------------------------------------
# Runtime mutation
# ------------------------------------------------------------------


def test_add_phrase_at_runtime(normalizer):
    """Runtime-added phrases should work."""
    normalizer.add_phrase("ticket book karo", "ticket book करो")
    text = "ticket book karo"
    result, norms = normalizer.normalize(text, target_language="hi")
    assert "करो" in result


def test_remove_phrase(normalizer):
    """Removed phrases should no longer apply."""
    normalizer.remove_phrase("bill print karo")
    text = "bill print karo"
    result, norms = normalizer.normalize(text, target_language="hi")
    # After removal the phrase-level normalisation should not fire
    assert not any("phrase: bill print karo" in n for n in norms)


# ------------------------------------------------------------------
# Custom phrases via constructor
# ------------------------------------------------------------------


def test_custom_phrases():
    """Custom phrases supplied at init should merge with defaults."""
    custom = {"status check karo": "status check करो"}
    norm = CodeSwitchNormalizer(enabled=True, custom_phrases=custom)
    text = "status check karo"
    result, norms = norm.normalize(text, target_language="hi")
    assert "करो" in result
