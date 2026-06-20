"""
SubtitleGenerator — generate SRT/VTT subtitle files from dubbed segments,
and optionally burn them into the output video via ffmpeg.
"""
import asyncio
from pathlib import Path
import structlog

from vaaniflow.models import AudioSegment
from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


def _format_srt_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_vtt_timestamp(ms: float) -> str:
    total_seconds = int(ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


class SubtitleGenerator:
    """Generates SRT and WebVTT subtitle files, and burns subtitles into video."""

    def __init__(self, enabled: bool = True, output_dir: str = "outputs"):
        self.enabled = enabled
        self.output_dir = Path(output_dir)

    def generate_srt(self, segments: list[AudioSegment], job_id: str, use_translated: bool = True) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        srt_path = self.output_dir / f"{job_id}.srt"

        lines = []
        for i, seg in enumerate(segments, start=1):
            text = (seg.translated_text if use_translated else seg.original_text) or ""
            lines.append(f"{i}")
            lines.append(f"{_format_srt_timestamp(seg.start_ms)} --> {_format_srt_timestamp(seg.end_ms)}")
            lines.append(text)
            lines.append("")

        srt_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("srt_generated", path=str(srt_path), segments=len(segments))
        return srt_path

    def generate_vtt(self, segments: list[AudioSegment], job_id: str, use_translated: bool = True) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vtt_path = self.output_dir / f"{job_id}.vtt"

        lines = ["WEBVTT", ""]
        for seg in segments:
            text = (seg.translated_text if use_translated else seg.original_text) or ""
            lines.append(f"{_format_vtt_timestamp(seg.start_ms)} --> {_format_vtt_timestamp(seg.end_ms)}")
            lines.append(text)
            lines.append("")

        vtt_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("vtt_generated", path=str(vtt_path), segments=len(segments))
        return vtt_path

    async def burn_subtitles(self, video_path: Path, srt_path: Path, output_path: Path) -> Path:
        if not video_path.exists():
            raise AudioProcessingError(f"Video file not found: {video_path}")
        if not srt_path.exists():
            raise AudioProcessingError(f"Subtitle file not found: {srt_path}")

        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"subtitles='{srt_escaped}'",
               "-c:a", "copy", str(output_path)]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise AudioProcessingError(f"Subtitle burn-in failed: {stderr.decode()}")

        log.info("subtitles_burned", output=str(output_path))
        return output_path
