"""
Segment timing utilities for adjusting TTS audio
to match original segment durations.
"""
import structlog

log = structlog.get_logger(__name__)


def calculate_speed_factor(
    original_duration_ms: float,
    tts_duration_ms: float,
    min_factor: float = 0.7,
    max_factor: float = 1.5,
) -> float:
    """
    Calculate the speed factor needed to match TTS audio
    to the original segment duration.

    Args:
        original_duration_ms: Duration of the original speech segment.
        tts_duration_ms: Duration of the synthesized TTS audio.
        min_factor: Minimum allowed speed (0.7x = slower).
        max_factor: Maximum allowed speed (1.5x = faster).

    Returns:
        Speed factor clamped to [min_factor, max_factor].
    """
    if tts_duration_ms <= 0:
        return 1.0

    factor = tts_duration_ms / original_duration_ms
    clamped = max(min_factor, min(max_factor, factor))

    if clamped != factor:
        log.warning(
            "speed_factor_clamped",
            original_factor=round(factor, 3),
            clamped_factor=round(clamped, 3),
            original_ms=original_duration_ms,
            tts_ms=tts_duration_ms,
        )

    return clamped


def calculate_silence_padding_ms(
    segment_start_ms: float,
    segment_end_ms: float,
    tts_duration_ms: float,
) -> float:
    """
    Calculate how much silence padding to insert after a TTS segment
    to maintain alignment with the original timing.

    Returns:
        Silence duration in milliseconds (non-negative).
    """
    expected_duration = segment_end_ms - segment_start_ms
    padding = expected_duration - tts_duration_ms
    return max(0.0, padding)


def calculate_gap_silence_ms(
    prev_end_ms: float,
    next_start_ms: float,
) -> float:
    """
    Calculate silence between two segments to preserve original gaps.

    Returns:
        Gap duration in milliseconds (non-negative).
    """
    gap = next_start_ms - prev_end_ms
    return max(0.0, gap)


def segments_overlap(
    seg1_start: float,
    seg1_end: float,
    seg2_start: float,
    seg2_end: float,
) -> bool:
    """Check if two time segments overlap."""
    return seg1_start < seg2_end and seg2_start < seg1_end
