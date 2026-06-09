"""
Audio stitching — reassemble TTS segments with original timing.
Uses pydub for audio manipulation.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional
import structlog

from vaaniflow.models import AudioSegment
from vaaniflow.exceptions import AudioProcessingError
from vaaniflow.utils.timing import calculate_silence_padding_ms, calculate_gap_silence_ms
from vaaniflow.config import settings

log = structlog.get_logger(__name__)


class AudioStitcher:
    """
    Stitch TTS audio segments back together with original timing.
    Inserts silence to match original segment gaps.
    """

    def __init__(self):
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def stitch(
        self,
        segments: list[AudioSegment],
        total_duration_ms: float,
        job_id: str,
    ) -> Path:
        """
        Stitch all TTS segments into a single audio file with original timing.

        Args:
            segments: List of AudioSegments with audio_bytes populated.
            total_duration_ms: Total duration of the original audio.
            job_id: Job ID for output filename.

        Returns:
            Path to the stitched output audio file.
        """
        try:
            from pydub import AudioSegment as PydubSegment
            from pydub.silence import detect_silence
        except ImportError:
            raise AudioProcessingError("pydub is required for audio stitching")

        log.info(
            "stitching_started",
            segments=len(segments),
            total_duration_ms=total_duration_ms,
            job_id=job_id,
        )

        try:
            # Start with empty audio
            final_audio = PydubSegment.silent(duration=0)
            current_position_ms = 0.0

            for i, segment in enumerate(segments):
                if segment.audio_bytes is None:
                    log.warning("segment_missing_audio", index=segment.index)
                    continue

                # Add gap silence before this segment
                gap_ms = segment.start_ms - current_position_ms
                if gap_ms > 0:
                    gap_silence = PydubSegment.silent(duration=int(gap_ms))
                    final_audio += gap_silence
                    log.debug("gap_silence_added", index=i, gap_ms=round(gap_ms, 1))

                # Load the TTS audio bytes
                try:
                    import io
                    tts_audio = PydubSegment.from_file(
                        io.BytesIO(segment.audio_bytes), format="mp3"
                    )
                except Exception:
                    # Try WAV format if MP3 fails
                    try:
                        tts_audio = PydubSegment.from_file(
                            io.BytesIO(segment.audio_bytes), format="wav"
                        )
                    except Exception as e:
                        log.warning(
                            "segment_audio_parse_failed",
                            index=i,
                            error=str(e),
                        )
                        # Insert silence for the segment duration
                        tts_audio = PydubSegment.silent(
                            duration=int(segment.duration_ms)
                        )

                # Adjust TTS audio to match original segment duration
                expected_duration_ms = segment.end_ms - segment.start_ms
                actual_duration_ms = len(tts_audio)

                if actual_duration_ms > expected_duration_ms * 1.1:
                    # TTS is too long — speed up (max 1.5x)
                    speed_factor = min(1.5, actual_duration_ms / expected_duration_ms)
                    tts_audio = tts_audio.speedup(playback_speed=speed_factor)
                    log.debug(
                        "segment_sped_up",
                        index=i,
                        factor=round(speed_factor, 2),
                    )

                final_audio += tts_audio
                current_position_ms = segment.end_ms

            # Pad to match original duration if needed
            if len(final_audio) < total_duration_ms:
                padding = PydubSegment.silent(
                    duration=int(total_duration_ms - len(final_audio))
                )
                final_audio += padding

            # Export to WAV
            output_path = self.output_dir / f"dubbed_{job_id}.wav"
            final_audio.export(str(output_path), format="wav")

            log.info(
                "stitching_completed",
                output_path=str(output_path),
                output_duration_ms=len(final_audio),
                output_size_bytes=output_path.stat().st_size,
            )

            return output_path

        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Audio stitching failed: {e}")
