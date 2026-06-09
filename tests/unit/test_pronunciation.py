"""
Unit tests for IndianNamePronunciationCorrector.
Tests lexicon substitution, case insensitivity, custom entries.
"""
import pytest
from vaaniflow.pronunciation.corrector import IndianNamePronunciationCorrector
from vaaniflow.pronunciation.indian_lexicon import INDIAN_PRONUNCIATION_MAP


@pytest.fixture
def corrector():
    return IndianNamePronunciationCorrector(enabled=True)


@pytest.fixture
def disabled_corrector():
    return IndianNamePronunciationCorrector(enabled=False)


def test_basic_correction(corrector):
    """Known Indian name should be corrected."""
    text = "I visited Bangalore last week"
    corrected, corrections = corrector.correct(text)
    assert "Baanga-lore" in corrected
    assert len(corrections) == 1
    assert "Bangalore" in corrections[0]


def test_case_insensitive(corrector):
    """Corrections should be case-insensitive."""
    text = "I love MUMBAI and mumbai"
    corrected, corrections = corrector.correct(text)
    assert "mumbai" not in corrected.lower() or "moom-bye" in corrected.lower()
    assert len(corrections) >= 1


def test_multiple_corrections(corrector):
    """Multiple Indian names in one text should all be corrected."""
    text = "Virat Kohli is from Mumbai and plays in Bangalore"
    corrected, corrections = corrector.correct(text)
    assert len(corrections) >= 3
    assert "Vee-rat Koh-lee" in corrected
    assert "Moom-bye" in corrected
    assert "Baanga-lore" in corrected


def test_no_correction_needed(corrector):
    """Text with no Indian names should be unchanged."""
    text = "The quick brown fox jumps over the lazy dog"
    corrected, corrections = corrector.correct(text)
    assert corrected == text
    assert len(corrections) == 0


def test_disabled_no_corrections(disabled_corrector):
    """Disabled corrector should return text unchanged."""
    text = "I visited Bangalore"
    corrected, corrections = disabled_corrector.correct(text)
    assert corrected == text
    assert len(corrections) == 0


def test_custom_map():
    """Custom pronunciation map should be merged with defaults."""
    custom = {"Flipkart": "Flip-kart"}
    corrector = IndianNamePronunciationCorrector(custom_map=custom)
    text = "Buy from Flipkart"
    corrected, corrections = corrector.correct(text)
    assert "Flip-kart" in corrected


def test_add_correction_at_runtime(corrector):
    """Runtime-added corrections should work."""
    corrector.add_correction("Meesho", "Mee-sho")
    text = "Order from Meesho"
    corrected, corrections = corrector.correct(text)
    assert "Mee-sho" in corrected


def test_remove_correction(corrector):
    """Removed corrections should no longer apply."""
    corrector.remove_correction("Bangalore")
    text = "I visited Bangalore"
    corrected, corrections = corrector.correct(text)
    assert "Bangalore" in corrected  # Should NOT be corrected anymore


def test_longest_match_first(corrector):
    """'Tamil Nadu' should match before 'Tamil' alone (if Tamil were a key)."""
    text = "Welcome to Tamil Nadu"
    corrected, corrections = corrector.correct(text)
    assert "Ta-mil Naa-doo" in corrected


def test_lexicon_has_entries():
    """Indian lexicon should have a reasonable number of entries."""
    assert len(INDIAN_PRONUNCIATION_MAP) >= 40


def test_word_boundary_only(corrector):
    """Should not match partial words."""
    text = "The Puneet was great"  # Should NOT match "Pune"
    corrected, corrections = corrector.correct(text)
    assert "Puneet" in corrected  # "Pune" correction should NOT apply here
