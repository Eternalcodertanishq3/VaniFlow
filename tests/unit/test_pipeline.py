"""
Unit tests for the dubbing pipeline.
Tests full pipeline flow with mocked providers.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from vaaniflow.models import (
    DubbingJob,
    DubbingJobConfig,
    JobStatus,
    SupportedLanguage,
    TTSProvider,
    TranslationProvider,
    TranscriptionProvider,
    TranscriptionResult,
    TranslationResult,
    TTSResult,
    AudioSegment,
)
from vaaniflow.providers.tts.base import TTSSynthesisResponse
from vaaniflow.exceptions import PipelineError


@pytest.fixture
def dubbing_job():
    """Create a sample dubbing job for testing."""
    config = DubbingJobConfig(
        source_language=SupportedLanguage.ENGLISH,
        target_language=SupportedLanguage.HINDI,
        tts_provider=TTSProvider.GTTS,
        translation_provider=TranslationProvider.GOOGLE,
    )
    return DubbingJob(config=config)


@pytest.fixture
def mock_transcription_result():
    return TranscriptionResult(
        segments=[
            AudioSegment(
                index=0, start_ms=0, end_ms=2000,
                duration_ms=2000, original_text="Hello world",
            ),
        ],
        source_language="en",
        total_duration_ms=2000,
        provider_used=TranscriptionProvider.WHISPER,
    )


@pytest.mark.asyncio
async def test_pipeline_status_transitions(dubbing_job):
    """Test that pipeline progresses through correct status stages."""
    assert dubbing_job.status == JobStatus.PENDING

    statuses_seen = []

    original_update = None

    async def track_status(job, status, progress):
        statuses_seen.append(status)
        job.status = status
        job.progress_pct = progress

    with patch("vaaniflow.pipeline.VaaniFlowPipeline._update_status", side_effect=track_status), \
         patch("vaaniflow.pipeline.VaaniFlowPipeline._transcribe") as mock_transcribe, \
         patch("vaaniflow.pipeline.VaaniFlowPipeline._translate") as mock_translate, \
         patch("vaaniflow.pipeline.VaaniFlowPipeline._synthesize") as mock_synthesize, \
         patch("vaaniflow.audio.extractor.AudioExtractor.extract") as mock_extract, \
         patch("vaaniflow.audio.stitcher.AudioStitcher.stitch") as mock_stitch:

        mock_extract.return_value = Path("/tmp/test.wav")
        mock_transcribe.return_value = TranscriptionResult(
            segments=[AudioSegment(index=0, start_ms=0, end_ms=1000, duration_ms=1000, original_text="test")],
            source_language="en", total_duration_ms=1000, provider_used=TranscriptionProvider.WHISPER,
        )
        mock_translate.return_value = TranslationResult(
            segments=[AudioSegment(index=0, start_ms=0, end_ms=1000, duration_ms=1000, original_text="test", translated_text="परीक्षा")],
            source_language=SupportedLanguage.ENGLISH, target_language=SupportedLanguage.HINDI,
            provider_used=TranslationProvider.GOOGLE,
        )
        mock_synthesize.return_value = TTSResult(
            segments=[AudioSegment(index=0, start_ms=0, end_ms=1000, duration_ms=1000, original_text="test", audio_bytes=b"audio")],
            provider_used=TTSProvider.GTTS, total_audio_bytes=5,
        )
        mock_stitch.return_value = Path("/tmp/output.wav")

        from vaaniflow.pipeline import VaaniFlowPipeline
        pipeline = VaaniFlowPipeline()
        await pipeline.run(dubbing_job, Path("/tmp/input.mp4"))

    expected_statuses = [
        JobStatus.TRANSCRIBING,   # stage 1
        JobStatus.TRANSCRIBING,   # stage 2
        JobStatus.TRANSLATING,    # stage 3
        JobStatus.SYNTHESIZING,   # stage 4
        JobStatus.STITCHING,      # stage 5
        JobStatus.COMPLETED,      # done
    ]
    assert statuses_seen == expected_statuses


@pytest.mark.asyncio
async def test_pipeline_sets_failed_on_error(dubbing_job):
    """Test that pipeline sets FAILED status on exception."""
    with patch("vaaniflow.audio.extractor.AudioExtractor.extract") as mock_extract:
        mock_extract.side_effect = Exception("ffmpeg not found")

        from vaaniflow.pipeline import VaaniFlowPipeline
        pipeline = VaaniFlowPipeline()

        with pytest.raises(PipelineError):
            await pipeline.run(dubbing_job, Path("/tmp/input.mp4"))

    assert dubbing_job.status == JobStatus.FAILED
    assert dubbing_job.error_message is not None


def test_dubbing_job_config_validation():
    """Test that source and target language must differ."""
    with pytest.raises(ValueError, match="Source and target language must be different"):
        DubbingJobConfig(
            source_language=SupportedLanguage.ENGLISH,
            target_language=SupportedLanguage.ENGLISH,
        )


def test_dubbing_job_config_defaults():
    """Test default config values."""
    config = DubbingJobConfig(target_language=SupportedLanguage.HINDI)
    assert config.source_language == SupportedLanguage.ENGLISH
    assert config.tts_provider == TTSProvider.SARVAM
    assert config.max_retries == 3
    assert config.timeout_seconds == 30
    assert config.preserve_timing is True


def test_dubbing_job_has_uuid():
    """Test that a new job gets a UUID."""
    config = DubbingJobConfig(target_language=SupportedLanguage.HINDI)
    job = DubbingJob(config=config)
    assert job.job_id is not None
    assert len(job.job_id) == 36  # UUID format
    assert job.status == JobStatus.PENDING
