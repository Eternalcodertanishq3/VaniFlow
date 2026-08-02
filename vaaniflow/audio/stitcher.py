"""
Audio stitching — reassemble TTS segments with original timing.
Uses native FFmpeg filtergraphs for highly efficient stream processing.
"""
import shutil
import tempfile
import asyncio
import subprocess
from pathlib import Path
import structlog

from vaaniflow.models import AudioSegment
from vaaniflow.exceptions import AudioProcessingError
from vaaniflow.config import settings

log = structlog.get_logger(__name__)


def _resolve_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _resolve_ffprobe() -> str | None:
    return shutil.which("ffprobe")


class AudioStitcher:
    """
    Stitch TTS audio segments back together with original timing.
    Uses FFmpeg filtergraphs to assemble audio efficiently without
    loading the entire uncompressed stream into Python's memory.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path(settings.output_dir)
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
        log.info(
            "stitching_started_ffmpeg",
            segments=len(segments),
            total_duration_ms=total_duration_ms,
            job_id=job_id,
        )

        output_path = self.output_dir / f"dubbed_{job_id}.wav"

        ffmpeg_path = _resolve_ffmpeg()
        if not ffmpeg_path:
            raise AudioProcessingError("ffmpeg not found in PATH")

        try:
            # Use a temporary directory to store individual segments for ffmpeg
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                inputs = []
                filter_parts = []
                concat_inputs = []

                current_position_ms = 0.0
                input_idx = 0

                for i, segment in enumerate(segments):
                    if not segment.audio_bytes:
                        log.warning("segment_missing_audio", index=segment.index)
                        continue

                    # Calculate and add gap silence before this segment
                    gap_ms = segment.start_ms - current_position_ms
                    if gap_ms > 0:
                        gap_sec = gap_ms / 1000.0
                        silence_label = f"silence{i}"
                        filter_parts.append(f"aevalsrc=exprs=0:d={gap_sec}:s=44100[{silence_label}]")
                        concat_inputs.append(f"[{silence_label}]")
                        log.debug("gap_silence_added", index=i, gap_ms=round(gap_ms, 1))

                    # Save TTS audio bytes to temp file
                    seg_file = temp_dir_path / f"seg_{i}.wav"
                    await asyncio.to_thread(seg_file.write_bytes, segment.audio_bytes)
                    inputs.append(str(seg_file))

                    # Get actual duration to determine if speedup is needed
                    duration_sec = await self._get_duration(seg_file)
                    actual_duration_ms = duration_sec * 1000.0
                    expected_duration_ms = segment.end_ms - segment.start_ms

                    audio_label = f"audio{i}"
                    # Apply atempo filter if TTS duration exceeds original segment space by >10%
                    if expected_duration_ms > 0 and actual_duration_ms > expected_duration_ms * 1.1:
                        speed_factor = min(1.5, actual_duration_ms / expected_duration_ms)
                        filter_parts.append(
                            f"[{input_idx}:a]atempo={speed_factor},"
                            f"aformat=sample_rates=44100:channel_layouts=mono[{audio_label}]"
                        )
                        log.debug("segment_sped_up", index=i, factor=round(speed_factor, 2))
                    else:
                        filter_parts.append(
                            f"[{input_idx}:a]aformat=sample_rates=44100:channel_layouts=mono[{audio_label}]"
                        )

                    concat_inputs.append(f"[{audio_label}]")
                    current_position_ms = segment.end_ms
                    input_idx += 1

                # Add final padding to match total duration
                if total_duration_ms > current_position_ms:
                    pad_ms = total_duration_ms - current_position_ms
                    pad_sec = pad_ms / 1000.0
                    filter_parts.append(f"aevalsrc=exprs=0:d={pad_sec}:s=44100[padding]")
                    concat_inputs.append("[padding]")

                if not concat_inputs:
                    # Fallback if no valid segments
                    total_sec = total_duration_ms / 1000.0
                    filter_parts.append(f"aevalsrc=exprs=0:d={total_sec}:s=44100[outa]")
                    concat_inputs = ["[outa]"]
                else:
                    concat_filter = "".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[outa]"
                    filter_parts.append(concat_filter)

                filtergraph = ";".join(filter_parts)

                # Construct FFmpeg command using full resolved executable path
                cmd = [ffmpeg_path, "-y"]
                for inp in inputs:
                    cmd.extend(["-i", inp])

                cmd.extend([
                    "-filter_complex", filtergraph,
                    "-map", "[outa]",
                    "-ac", "1",
                    "-ar", "44100",
                    str(output_path)
                ])

                log.debug("ffmpeg_stitching_cmd", command=" ".join(cmd))

                def _run_stitch():
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                    if res.returncode != 0:
                        err = res.stderr.decode(errors="replace") if res.stderr else "Unknown error"
                        log.error("ffmpeg_stitching_failed", stderr=err[:500])
                        raise AudioProcessingError(f"FFmpeg stitching failed: {err[:500]}")

                await asyncio.to_thread(_run_stitch)

                # Output validation
                if not output_path.exists():
                    raise AudioProcessingError("FFmpeg completed but output file is missing")

                log.info(
                    "stitching_completed",
                    output_path=str(output_path),
                    output_size_bytes=output_path.stat().st_size,
                )

                return output_path

        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Audio stitching failed ({type(e).__name__}): {e}")

    async def merge_video_and_audio(
        self, original_video_path: Path, dubbed_audio_path: Path, job_id: str
    ) -> Path:
        """Combine original video stream with new dubbed audio track into a video file."""
        output_video_path = self.output_dir / f"dubbed_video_{job_id}{original_video_path.suffix}"
        ffmpeg_path = _resolve_ffmpeg()
        if not ffmpeg_path or not original_video_path.exists():
            return dubbed_audio_path

        def _run_merge():
            cmd = [
                ffmpeg_path, "-y",
                "-i", str(original_video_path),
                "-i", str(dubbed_audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_video_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if res.returncode == 0 and output_video_path.exists():
                log.info("video_audio_merged", output=str(output_video_path))
                return output_video_path
            return dubbed_audio_path

        return await asyncio.to_thread(_run_merge)

    async def _get_duration(self, file_path: Path) -> float:
        """Get audio file duration using ffprobe."""
        ffprobe_path = _resolve_ffprobe()
        if not ffprobe_path:
            return 0.0

        def _run_ffprobe():
            cmd = [
                ffprobe_path, "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", str(file_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode == 0:
                try:
                    return float(res.stdout.decode().strip())
                except ValueError:
                    pass
            return 0.0

        return await asyncio.to_thread(_run_ffprobe)
