"""
Pytest fixtures. All external providers must be mocked in tests.
Sarvam JD: "Experience writing tests and mocking external services"
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from api.main import app
from vaaniflow.models import (
    AudioSegment,
    SupportedLanguage,
    TranscriptionProvider,
    TranscriptionResult,
)
from vaaniflow.providers.tts.base import TTSSynthesisResponse


@pytest.fixture
def mock_elevenlabs_provider():
    """Mock ElevenLabs to return dummy audio bytes."""
    with patch(
        "vaaniflow.providers.tts.elevenlabs_provider.ElevenLabsProvider.synthesize"
    ) as mock:
        mock.return_value = TTSSynthesisResponse(
            audio_bytes=b"fake_audio_bytes_" * 100,
            duration_ms=2000.0,
            provider="elevenlabs",
        )
        yield mock


@pytest.fixture
def mock_google_translate():
    """Mock Google Translate API."""
    with patch(
        "vaaniflow.providers.translation.google_provider.GoogleTranslationProvider.translate"
    ) as mock:
        mock.return_value = "नमस्ते दुनिया"  # "Hello World" in Hindi
        yield mock


@pytest.fixture
def mock_whisper():
    """Mock Whisper transcription with realistic segments."""
    with patch(
        "vaaniflow.providers.transcription.whisper_provider.WhisperProvider.transcribe"
    ) as mock:
        mock.return_value = TranscriptionResult(
            segments=[
                AudioSegment(
                    index=0,
                    start_ms=0,
                    end_ms=2000,
                    duration_ms=2000,
                    original_text="Hello world",
                ),
                AudioSegment(
                    index=1,
                    start_ms=2500,
                    end_ms=5000,
                    duration_ms=2500,
                    original_text="This is a test",
                ),
            ],
            source_language="en",
            total_duration_ms=5000,
            provider_used=TranscriptionProvider.WHISPER,
        )
        yield mock


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache — always return None (cache miss)."""
    with patch("vaaniflow.cache.redis_cache.TranslationCache.get") as mock_get, \
         patch("vaaniflow.cache.redis_cache.TranslationCache.set") as mock_set:
        mock_get.return_value = None
        mock_set.return_value = True
        yield mock_get, mock_set


@pytest_asyncio.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_audio_segment():
    """Create a sample AudioSegment for testing."""
    return AudioSegment(
        index=0,
        start_ms=0,
        end_ms=2000,
        duration_ms=2000,
        original_text="Hello world",
        translated_text="नमस्ते दुनिया",
    )


@pytest.fixture
def sample_transcription_result():
    """Create a sample TranscriptionResult for testing."""
    return TranscriptionResult(
        segments=[
            AudioSegment(
                index=0,
                start_ms=0,
                end_ms=2000,
                duration_ms=2000,
                original_text="Hello world",
            ),
            AudioSegment(
                index=1,
                start_ms=2500,
                end_ms=5000,
                duration_ms=2500,
                original_text="This is a test",
            ),
            AudioSegment(
                index=2,
                start_ms=5500,
                end_ms=8000,
                duration_ms=2500,
                original_text="How are you doing",
            ),
        ],
        source_language="en",
        total_duration_ms=8000,
        provider_used=TranscriptionProvider.WHISPER,
    )
