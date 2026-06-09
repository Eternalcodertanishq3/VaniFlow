"""
Unit tests for AmbientAudioPreserver.
Tests separation and remix with mocked scipy.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from vaaniflow.audio.ambient_separator import AmbientAudioPreserver, SeparationResult


@pytest.fixture
def preserver():
    return AmbientAudioPreserver(enabled=True)


@pytest.fixture
def disabled_preserver():
    return AmbientAudioPreserver(enabled=False)


@pytest.mark.asyncio
async def test_disabled_returns_original(disabled_preserver):
    """Disabled preserver should return original audio unchanged."""
    audio = b"original_audio_bytes" * 100
    result = await disabled_preserver.separate(audio)
    assert result.vocals_bytes == audio
    assert result.ambient_bytes == b""
    assert result.has_significant_ambient is False


@pytest.mark.asyncio
async def test_scipy_unavailable_returns_original(preserver):
    """If scipy is not installed, return original audio."""
    preserver._scipy_available = False
    audio = b"original_audio" * 100
    result = await preserver.separate(audio)
    assert result.vocals_bytes == audio
    assert result.ambient_bytes == b""


@pytest.mark.asyncio
async def test_separation_result_structure():
    """SeparationResult should have correct fields."""
    result = SeparationResult(
        vocals_bytes=b"vocals",
        ambient_bytes=b"ambient",
        ambient_level_db=-20.0,
        has_significant_ambient=True,
    )
    assert result.vocals_bytes == b"vocals"
    assert result.ambient_bytes == b"ambient"
    assert result.ambient_level_db == -20.0
    assert result.has_significant_ambient is True


@pytest.mark.asyncio
async def test_remix_no_ambient_returns_dubbed(preserver):
    """Remix with empty ambient should return dubbed audio unchanged."""
    dubbed = b"dubbed_audio_bytes"
    result = await preserver.remix(dubbed, b"")
    assert result == dubbed


@pytest.mark.asyncio
async def test_remix_disabled_returns_dubbed(disabled_preserver):
    """Disabled preserver remix should return dubbed audio unchanged."""
    dubbed = b"dubbed_audio_bytes"
    result = await disabled_preserver.remix(dubbed, b"ambient")
    assert result == dubbed


@pytest.mark.asyncio
async def test_separation_error_graceful(preserver):
    """Separation errors should fall back gracefully."""
    preserver._scipy_available = True
    with patch.object(preserver, '_separate_sync', side_effect=Exception("STFT failed")):
        result = await preserver.separate(b"audio" * 100)
        assert result.vocals_bytes == b"audio" * 100
        assert result.ambient_bytes == b""
