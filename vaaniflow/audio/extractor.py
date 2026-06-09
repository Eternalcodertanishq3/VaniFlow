"""
Extract audio track from video files using ffmpeg.
"""
import asyncio
import tempfile
from pathlib import Path
import structlog

from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


class AudioExtractor:
    """
    Extract audio from video/audio files using ffmpeg.
    Outputs normalized WAV for downstream processing.
    """

    async def extract(self, input_path: Path) -> Path:
        """
        Extract audio from a video/audio file.

        Args:
            input_path: Path to input video/audio file.

        Returns:
            Path to extracted WAV audio file.

        Raises:
            AudioProcessingError on failure.
        """
        if not input_path.exists():
            raise AudioProcessingError(f"Input file not found: {input_path}")

        # Output to temp WAV file
        output_path = Path(tempfile.mktemp(suffix=".wav"))

        log.info(
            "audio_extraction_started",
            input=str(input_path),
            output=str(output_path),
        )

        try:
            # Use ffmpeg to extract audio as 16kHz mono WAV
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-vn",                    # no video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", "16000",           # 16kHz sample rate
                "-ac", "1",               # mono
                "-y",                     # overwrite
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                raise AudioProcessingError(
                    f"ffmpeg extraction failed (code {process.returncode}): {error_msg}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioProcessingError("ffmpeg produced empty output file")

            log.info(
                "audio_extraction_completed",
                output=str(output_path),
                size_bytes=output_path.stat().st_size,
            )

            return output_path

        except AudioProcessingError:
            raise
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffmpeg not found. Install ffmpeg: https://ffmpeg.org/download.html"
            )
        except Exception as e:
            raise AudioProcessingError(f"Audio extraction failed: {e}")

    async def get_duration_ms(self, audio_path: Path) -> float:
        """Get duration of an audio file in milliseconds using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise AudioProcessingError("ffprobe failed")

            duration_s = float(stdout.decode().strip())
            return duration_s * 1000

        except (ValueError, AudioProcessingError) as e:
            raise AudioProcessingError(f"Failed to get audio duration: {e}")
