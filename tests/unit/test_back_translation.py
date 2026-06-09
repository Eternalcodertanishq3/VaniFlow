"""
Unit tests for BackTranslationQualityScorer.
Tests BLEU scoring, threshold pass/fail, short-text skip.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from vaaniflow.quality.back_translation import (
    BackTranslationQualityScorer, BackTranslationScore,
)


@pytest.fixture
def scorer():
    return BackTranslationQualityScorer(threshold=0.30, enabled=True)


@pytest.fixture
def disabled_scorer():
    return BackTranslationQualityScorer(threshold=0.30, enabled=False)


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.translate = AsyncMock(return_value="Hello how are you doing today")
    return provider


@pytest.mark.asyncio
async def test_disabled_returns_pass(disabled_scorer, mock_provider):
    """Disabled scorer should always return passed=True."""
    result = await disabled_scorer.score(
        original_text="Hello how are you doing today",
        translated_text="नमस्ते आप कैसे हैं आज",
        source_lang="en", target_lang="hi",
        translation_provider=mock_provider,
    )
    assert result.passed is True
    assert result.bleu_score == 1.0
    assert result.should_retry is False


@pytest.mark.asyncio
async def test_short_text_skipped(scorer, mock_provider):
    """Text with <4 words should be skipped."""
    result = await scorer.score(
        original_text="Hello world",
        translated_text="नमस्ते दुनिया",
        source_lang="en", target_lang="hi",
        translation_provider=mock_provider,
    )
    assert result.passed is True
    assert result.bleu_score == 1.0


@pytest.mark.asyncio
async def test_good_back_translation_passes(scorer, mock_provider):
    """Good back-translation (high BLEU) should pass."""
    original = "Hello how are you doing today"
    mock_provider.translate.return_value = "Hello how are you doing today"

    result = await scorer.score(
        original_text=original,
        translated_text="नमस्ते आप कैसे हैं आज",
        source_lang="en", target_lang="hi",
        translation_provider=mock_provider,
    )
    assert result.passed is True
    assert result.bleu_score > 0.3


@pytest.mark.asyncio
async def test_bad_back_translation_fails(scorer, mock_provider):
    """Completely wrong back-translation should fail."""
    mock_provider.translate.return_value = "The weather is sunny and bright"

    result = await scorer.score(
        original_text="Hello how are you doing today my friend",
        translated_text="नमस्ते आप कैसे हैं",
        source_lang="en", target_lang="hi",
        translation_provider=mock_provider,
    )
    assert result.bleu_score < 0.3
    assert result.should_retry is True


@pytest.mark.asyncio
async def test_provider_error_graceful(scorer):
    """Provider error during back-translation should return safe defaults."""
    error_provider = AsyncMock()
    error_provider.translate.side_effect = Exception("API down")

    result = await scorer.score(
        original_text="Hello how are you doing today",
        translated_text="नमस्ते आप कैसे हैं",
        source_lang="en", target_lang="hi",
        translation_provider=error_provider,
    )
    assert result.passed is True
    assert result.bleu_score == 0.5


def test_bleu_sync_identical(scorer):
    """Identical text should have high BLEU score."""
    score = scorer._bleu_sync(
        "the quick brown fox jumps over",
        "the quick brown fox jumps over",
    )
    assert score > 0.9


def test_bleu_sync_different(scorer):
    """Completely different text should have low BLEU."""
    score = scorer._bleu_sync(
        "the quick brown fox jumps over",
        "sunny weather is nice today",
    )
    assert score < 0.3


def test_bleu_sync_empty(scorer):
    """Empty strings should return 0."""
    assert scorer._bleu_sync("", "hello") == 0.0
    assert scorer._bleu_sync("hello", "") == 0.0
