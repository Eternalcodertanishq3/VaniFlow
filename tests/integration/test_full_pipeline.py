"""
End-to-end integration test for the full dubbing pipeline.
All external services are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from vaaniflow.models import (
    DubbingJob,
    DubbingJobConfig,
    SupportedLanguage,
    TTSProvider,
    TranslationProvider,
    TranscriptionProvider,
    TranscriptionResult,
    AudioSegment,
    JobStatus,
)
from vaaniflow.providers.tts.base import TTSSynthesisResponse
from vaaniflow.pipeline import VaaniFlowPipeline


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_providers():
    """
    End-to-end test: extract → transcribe → translate → synthesize → stitch.
    All external providers are mocked.
    """
    config = DubbingJobConfig(
        source_language=SupportedLanguage.ENGLISH,
        target_language=SupportedLanguage.HINDI,
        tts_provider=TTSProvider.GTTS,
        translation_provider=TranslationProvider.GOOGLE,
    )
    job = DubbingJob(config=config)

    mock_segments = [
        AudioSegment(
            index=0, start_ms=0, end_ms=2000,
            duration_ms=2000, original_text="Hello world",
        ),
        AudioSegment(
            index=1, start_ms=2500, end_ms=5000,
            duration_ms=2500, original_text="This is a test",
        ),
    ]

    mock_transcription = TranscriptionResult(
        segments=mock_segments,
        source_language="en",
        total_duration_ms=5000,
        provider_used=TranscriptionProvider.WHISPER,
    )

    with patch("vaaniflow.audio.extractor.AudioExtractor.extract") as mock_extract, \
         patch("vaaniflow.providers.transcription.whisper_provider.WhisperProvider.transcribe") as mock_transcribe, \
         patch("vaaniflow.providers.translation.google_provider.GoogleTranslationProvider.translate") as mock_translate, \
         patch("vaaniflow.providers.tts.gtts_provider.GTTSProvider.synthesize") as mock_synthesize, \
         patch("vaaniflow.providers.tts.gtts_provider.GTTSProvider.synthesize_with_logging") as mock_synth_log, \
         patch("vaaniflow.audio.stitcher.AudioStitcher.stitch") as mock_stitch, \
         patch("vaaniflow.cache.redis_cache.TranslationCache.get") as mock_cache_get, \
         patch("vaaniflow.cache.redis_cache.TranslationCache.set") as mock_cache_set:

        # Configure mocks
        mock_extract.return_value = Path("/tmp/extracted.wav")
        mock_transcribe.return_value = mock_transcription
        mock_translate.return_value = "नमस्ते दुनिया"
        mock_synth_log.return_value = TTSSynthesisResponse(
            audio_bytes=b"fake_audio_" * 50,
            duration_ms=2000.0,
            provider="gtts",
        )
        mock_stitch.return_value = Path("/tmp/output.wav")
        mock_cache_get.return_value = None
        mock_cache_set.return_value = True

        pipeline = VaaniFlowPipeline()
        output_path = await pipeline.run(job, Path("/tmp/input.mp4"))

        # Verify pipeline completed
        assert job.status == JobStatus.COMPLETED
        assert job.progress_pct == 100.0
        assert output_path == Path("/tmp/output.wav")

        # Verify all stages were called
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        assert mock_translate.call_count == 2  # Two segments
        assert mock_synth_log.call_count == 2  # Two segments synthesized concurrently
        mock_stitch.assert_called_once()

        # Verify cache was populated
        assert mock_cache_set.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_uses_cache_hits():
    """Test that pipeline uses cached translations when available."""
    config = DubbingJobConfig(
        source_language=SupportedLanguage.ENGLISH,
        target_language=SupportedLanguage.TAMIL,
        tts_provider=TTSProvider.GTTS,
        translation_provider=TranslationProvider.GOOGLE,
    )
    job = DubbingJob(config=config)

    mock_transcription = TranscriptionResult(
        segments=[
            AudioSegment(index=0, start_ms=0, end_ms=2000, duration_ms=2000, original_text="Hello"),
        ],
        source_language="en",
        total_duration_ms=2000,
        provider_used=TranscriptionProvider.WHISPER,
    )

    with patch("vaaniflow.audio.extractor.AudioExtractor.extract") as mock_extract, \
         patch("vaaniflow.providers.transcription.whisper_provider.WhisperProvider.transcribe") as mock_transcribe, \
         patch("vaaniflow.providers.translation.google_provider.GoogleTranslationProvider.translate") as mock_translate, \
         patch("vaaniflow.providers.tts.gtts_provider.GTTSProvider.synthesize_with_logging") as mock_synth, \
         patch("vaaniflow.audio.stitcher.AudioStitcher.stitch") as mock_stitch, \
         patch("vaaniflow.cache.redis_cache.TranslationCache.get") as mock_cache_get, \
         patch("vaaniflow.cache.redis_cache.TranslationCache.set") as mock_cache_set:

        mock_extract.return_value = Path("/tmp/extracted.wav")
        mock_transcribe.return_value = mock_transcription
        # Cache HIT — translate should NOT be called
        mock_cache_get.return_value = "வணக்கம்"
        mock_synth.return_value = TTSSynthesisResponse(
            audio_bytes=b"audio" * 50, duration_ms=1500, provider="gtts",
        )
        mock_stitch.return_value = Path("/tmp/output.wav")

        pipeline = VaaniFlowPipeline()
        await pipeline.run(job, Path("/tmp/input.mp4"))

        # Translation provider should NOT have been called (cache hit)
        mock_translate.assert_not_called()
        # Cache set should NOT be called (already cached)
        mock_cache_set.assert_not_called()
