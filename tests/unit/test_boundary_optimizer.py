"""
Unit tests for SmartSegmentBoundaryOptimizer.
Tests segment merging, gap constraint, word limit.
"""
import pytest
from unittest.mock import patch, MagicMock
from vaaniflow.segmentation.boundary_optimizer import SmartSegmentBoundaryOptimizer
from vaaniflow.models import AudioSegment, TranscriptionResult, TranscriptionProvider


@pytest.fixture
def optimizer():
    return SmartSegmentBoundaryOptimizer(enabled=True)


@pytest.fixture
def disabled_optimizer():
    return SmartSegmentBoundaryOptimizer(enabled=False)


@pytest.fixture
def fragmented_transcription():
    """Two segments that form one complete sentence."""
    return TranscriptionResult(
        segments=[
            AudioSegment(
                index=0, start_ms=0, end_ms=1500, duration_ms=1500,
                original_text="The quick brown fox",
            ),
            AudioSegment(
                index=1, start_ms=1600, end_ms=3000, duration_ms=1400,
                original_text="jumped over the lazy dog.",
            ),
        ],
        source_language="en",
        total_duration_ms=3000,
        provider_used=TranscriptionProvider.WHISPER,
    )


@pytest.fixture
def complete_sentences_transcription():
    """Two segments that are complete sentences — should NOT merge."""
    return TranscriptionResult(
        segments=[
            AudioSegment(
                index=0, start_ms=0, end_ms=2000, duration_ms=2000,
                original_text="Hello world.",
            ),
            AudioSegment(
                index=1, start_ms=2500, end_ms=5000, duration_ms=2500,
                original_text="This is a test.",
            ),
        ],
        source_language="en",
        total_duration_ms=5000,
        provider_used=TranscriptionProvider.WHISPER,
    )


@pytest.fixture
def large_gap_transcription():
    """Two segments with a gap >800ms — should NOT merge."""
    return TranscriptionResult(
        segments=[
            AudioSegment(
                index=0, start_ms=0, end_ms=1000, duration_ms=1000,
                original_text="First part",
            ),
            AudioSegment(
                index=1, start_ms=2500, end_ms=4000, duration_ms=1500,
                original_text="second part after long pause",
            ),
        ],
        source_language="en",
        total_duration_ms=4000,
        provider_used=TranscriptionProvider.WHISPER,
    )


@pytest.mark.asyncio
async def test_disabled_returns_unchanged(disabled_optimizer, fragmented_transcription):
    """Disabled optimizer should return input unchanged."""
    result = await disabled_optimizer.optimize(fragmented_transcription)
    assert len(result.segments) == 2


@pytest.mark.asyncio
async def test_single_segment_unchanged(optimizer):
    """Single segment should be returned unchanged."""
    transcription = TranscriptionResult(
        segments=[AudioSegment(
            index=0, start_ms=0, end_ms=2000, duration_ms=2000,
            original_text="Hello world.",
        )],
        source_language="en", total_duration_ms=2000,
        provider_used=TranscriptionProvider.WHISPER,
    )
    result = await optimizer.optimize(transcription)
    assert len(result.segments) == 1


@pytest.mark.asyncio
async def test_large_gap_prevents_merge(optimizer, large_gap_transcription):
    """Segments with >800ms gap should NOT be merged."""
    result = await optimizer.optimize(large_gap_transcription)
    assert len(result.segments) == 2


@pytest.mark.asyncio
async def test_spacy_unavailable_returns_unchanged(fragmented_transcription):
    """If spaCy is not installed, return unchanged."""
    optimizer = SmartSegmentBoundaryOptimizer(enabled=True)
    optimizer._nlp = None
    with patch.object(optimizer, '_get_nlp', return_value=None):
        result = await optimizer.optimize(fragmented_transcription)
        assert len(result.segments) == 2


def test_merge_segments_preserves_timing(optimizer):
    """Merged segment should span full time range."""
    segments = [
        AudioSegment(index=0, start_ms=0, end_ms=1500, duration_ms=1500, original_text="Hello"),
        AudioSegment(index=1, start_ms=1600, end_ms=3000, duration_ms=1400, original_text="world"),
    ]
    merged = optimizer._merge_segments(segments, 0)
    assert merged.start_ms == 0
    assert merged.end_ms == 3000
    assert merged.duration_ms == 3000
    assert merged.original_text == "Hello world"
    assert merged.index == 0
