"""
Unit tests for EmotionPreserver.
Tests neutral fallback, emotion classification, and TTS param mapping.
"""
import pytest
from unittest.mock import patch, MagicMock
from vaaniflow.emotion.detector import (
    EmotionPreserver, EmotionLabel, EmotionResult, EMOTION_TTS_PARAMS,
)


@pytest.fixture
def preserver():
    return EmotionPreserver(enabled=True)


@pytest.fixture
def disabled_preserver():
    return EmotionPreserver(enabled=False)


@pytest.mark.asyncio
async def test_disabled_returns_neutral(disabled_preserver):
    """Disabled preserver should always return neutral."""
    result = await disabled_preserver.detect(b"fake_audio_data" * 100)
    assert result.label == EmotionLabel.NEUTRAL
    assert result.confidence == 1.0
    assert result.speaking_rate == 1.0
    assert result.pitch_shift == 0.0


@pytest.mark.asyncio
async def test_short_audio_returns_neutral(preserver):
    """Audio <500 bytes should return neutral."""
    result = await preserver.detect(b"short")
    assert result.label == EmotionLabel.NEUTRAL


@pytest.mark.asyncio
async def test_empty_audio_returns_neutral(preserver):
    """Empty audio should return neutral."""
    result = await preserver.detect(b"")
    assert result.label == EmotionLabel.NEUTRAL


@pytest.mark.asyncio
async def test_librosa_unavailable_returns_neutral(preserver):
    """If librosa is not installed, return neutral gracefully."""
    preserver._librosa_available = False
    result = await preserver.detect(b"fake_audio" * 100)
    assert result.label == EmotionLabel.NEUTRAL


def test_emotion_tts_params_complete():
    """All emotion labels should have TTS parameter mappings."""
    for label in EmotionLabel:
        assert label in EMOTION_TTS_PARAMS
        params = EMOTION_TTS_PARAMS[label]
        assert "speaking_rate" in params
        assert "pitch_shift" in params
        assert "stability" in params


def test_classify_emotion_excited(preserver):
    """High energy + fast tempo + high pitch → excited."""
    result = preserver._classify_emotion(
        pitch_mean=250, energy_rms=0.12,
        tempo=180, spectral_centroid=3500, zcr=0.1,
    )
    assert result == EmotionLabel.EXCITED


def test_classify_emotion_sad(preserver):
    """Low pitch + slow tempo + low energy → sad."""
    result = preserver._classify_emotion(
        pitch_mean=100, energy_rms=0.01,
        tempo=70, spectral_centroid=1500, zcr=0.02,
    )
    assert result == EmotionLabel.SAD


def test_classify_emotion_neutral(preserver):
    """Mid-range features → neutral."""
    result = preserver._classify_emotion(
        pitch_mean=170, energy_rms=0.05,
        tempo=120, spectral_centroid=2000, zcr=0.05,
    )
    assert result == EmotionLabel.NEUTRAL


def test_neutral_result_structure(preserver):
    """Neutral result should have correct structure."""
    result = preserver._neutral_result()
    assert isinstance(result, EmotionResult)
    assert result.label == EmotionLabel.NEUTRAL
    assert result.confidence == 1.0
    assert result.pitch_mean_hz == 0.0
    assert result.energy_rms == 0.0
    assert result.tempo_bpm == 0.0
