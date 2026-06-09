"""
Unit tests for the Quality Control pipeline.
Tests segment validation: silence ratio, length ratio, min bytes.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from vaaniflow.qc.pipeline import QualityController
from vaaniflow.qc.models import QCStatus, QCConfig, SegmentQCResult, PipelineQCResult
from vaaniflow.models import AudioSegment


@pytest.fixture
def qc_controller():
    config = QCConfig(
        max_silence_ratio=0.7,
        max_length_ratio=3.0,
        min_length_ratio=0.3,
        min_audio_bytes=100,
    )
    return QualityController(config=config)


@pytest.fixture
def good_segment():
    return AudioSegment(
        index=0, start_ms=0, end_ms=2000, duration_ms=2000,
        original_text="Hello world",
        audio_bytes=b"x" * 500,
    )


@pytest.fixture
def tiny_segment():
    return AudioSegment(
        index=1, start_ms=0, end_ms=2000, duration_ms=2000,
        original_text="Hi",
        audio_bytes=b"x" * 10,  # Too small
    )


@pytest.fixture
def empty_segment():
    return AudioSegment(
        index=2, start_ms=0, end_ms=2000, duration_ms=2000,
        original_text="Empty",
        audio_bytes=None,
    )


@pytest.mark.asyncio
async def test_qc_too_small_audio_fails(qc_controller, tiny_segment):
    """Audio below min_audio_bytes should FAIL."""
    result = await qc_controller.validate_pipeline_output([tiny_segment])
    assert result.overall_status == QCStatus.FAIL
    assert result.fail_count == 1
    assert result.segments[0].should_retry is True
    assert "too small" in result.segments[0].issues[0].lower()


@pytest.mark.asyncio
async def test_qc_empty_audio_fails(qc_controller, empty_segment):
    """Segment with no audio bytes should FAIL."""
    result = await qc_controller.validate_pipeline_output([empty_segment])
    assert result.overall_status == QCStatus.FAIL
    assert result.fail_count == 1


@pytest.mark.asyncio
async def test_qc_good_audio_passes(qc_controller, good_segment):
    """Good audio with reasonable size should pass (mocking pydub)."""
    with patch("vaaniflow.qc.pipeline.QualityController._compute_silence_ratio", return_value=0.1), \
         patch("vaaniflow.qc.pipeline.QualityController._estimate_duration_ms", return_value=2000.0):
        result = await qc_controller.validate_pipeline_output([good_segment])
        assert result.overall_status == QCStatus.PASS
        assert result.pass_count == 1
        assert result.fail_count == 0


@pytest.mark.asyncio
async def test_qc_high_silence_ratio_fails(qc_controller, good_segment):
    """Segment with >70% silence should FAIL."""
    with patch("vaaniflow.qc.pipeline.QualityController._compute_silence_ratio", return_value=0.9):
        result = await qc_controller.validate_pipeline_output([good_segment])
        assert result.overall_status == QCStatus.FAIL
        assert result.segments[0].should_retry is True


@pytest.mark.asyncio
async def test_qc_long_tts_warns(qc_controller, good_segment):
    """TTS >3x original length should WARN."""
    with patch("vaaniflow.qc.pipeline.QualityController._compute_silence_ratio", return_value=0.1), \
         patch("vaaniflow.qc.pipeline.QualityController._estimate_duration_ms", return_value=8000.0):
        result = await qc_controller.validate_pipeline_output([good_segment])
        assert result.overall_status == QCStatus.WARN
        assert result.warn_count == 1


@pytest.mark.asyncio
async def test_qc_multiple_segments_mixed(qc_controller, good_segment, tiny_segment):
    """Mix of good and bad segments: overall FAIL if any fails."""
    with patch("vaaniflow.qc.pipeline.QualityController._compute_silence_ratio", return_value=0.1), \
         patch("vaaniflow.qc.pipeline.QualityController._estimate_duration_ms", return_value=2000.0):
        result = await qc_controller.validate_pipeline_output([good_segment, tiny_segment])
        assert result.overall_status == QCStatus.FAIL
        assert result.pass_count == 1
        assert result.fail_count == 1
        assert 1 in result.retry_segments


@pytest.mark.asyncio
async def test_qc_config_defaults():
    """QCConfig should have sensible defaults."""
    config = QCConfig()
    assert config.max_silence_ratio == 0.7
    assert config.max_length_ratio == 3.0
    assert config.min_length_ratio == 0.3
    assert config.min_audio_bytes == 100
    assert config.auto_retry_on_fail is True
