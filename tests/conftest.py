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


# ============ Phase 2 Fixtures ============

@pytest.fixture
def mock_emotion_preserver():
    """Mock emotion preserver to return neutral."""
    with patch("vaaniflow.emotion.detector.EmotionPreserver.detect") as mock:
        from vaaniflow.emotion.detector import EmotionResult, EmotionLabel
        mock.return_value = EmotionResult(
            label=EmotionLabel.NEUTRAL, confidence=1.0,
            pitch_mean_hz=0.0, energy_rms=0.0, tempo_bpm=0.0,
            speaking_rate=1.0, pitch_shift=0.0, tts_stability=0.75,
        )
        yield mock


@pytest.fixture
def mock_pronunciation_corrector():
    """Mock pronunciation corrector to return unchanged text."""
    with patch(
        "vaaniflow.pronunciation.corrector.IndianNamePronunciationCorrector.correct"
    ) as mock:
        mock.side_effect = lambda text: (text, [])
        yield mock


@pytest.fixture
def mock_boundary_optimizer():
    """Mock boundary optimizer to return unchanged transcription."""
    with patch(
        "vaaniflow.segmentation.boundary_optimizer.SmartSegmentBoundaryOptimizer.optimize"
    ) as mock:
        mock.side_effect = lambda t: t
        yield mock


@pytest.fixture
def mock_ambient_preserver():
    """Mock ambient preserver to skip separation."""
    with patch(
        "vaaniflow.audio.ambient_separator.AmbientAudioPreserver.separate"
    ) as mock_sep, patch(
        "vaaniflow.audio.ambient_separator.AmbientAudioPreserver.remix"
    ) as mock_remix:
        from vaaniflow.audio.ambient_separator import SeparationResult
        mock_sep.return_value = SeparationResult(
            vocals_bytes=b"", ambient_bytes=b"",
            ambient_level_db=-96.0, has_significant_ambient=False,
        )
        mock_remix.side_effect = lambda dubbed, ambient: dubbed
        yield mock_sep, mock_remix


@pytest.fixture
def mock_qc_controller():
    """Mock QC controller to pass everything."""
    with patch(
        "vaaniflow.qc.pipeline.QualityController.validate_pipeline_output"
    ) as mock:
        from vaaniflow.qc.models import PipelineQCResult, QCStatus
        mock.return_value = PipelineQCResult(
            overall_status=QCStatus.PASS,
            segments=[], pass_count=2, warn_count=0,
            fail_count=0, retry_segments=[],
        )
        yield mock


@pytest.fixture
def mock_back_translation_scorer():
    """Mock back-translation scorer to always pass."""
    with patch(
        "vaaniflow.quality.back_translation.BackTranslationQualityScorer.score"
    ) as mock:
        from vaaniflow.quality.back_translation import BackTranslationScore
        mock.return_value = BackTranslationScore(
            original_text="", translated_text="",
            back_translated_text="", bleu_score=0.8,
            passed=True, should_retry=False,
        )
        yield mock


@pytest.fixture
def mock_job_repo():
    """Mock DubbingJobRepository for API tests."""
    with patch("api.routes.jobs.job_repo") as mock:
        mock.save = AsyncMock(return_value=True)
        mock.get = AsyncMock(return_value=None)
        mock.list_all = AsyncMock(return_value=[])
        mock.delete = AsyncMock(return_value=True)
        yield mock
