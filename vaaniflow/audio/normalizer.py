"""
Audio normalization utilities.
Volume leveling and sample rate conversion.
"""
import asyncio
import shutil
import subprocess
from pathlib import Path
import structlog

from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


def _resolve_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


class AudioNormalizer:
    """
    Audio normalization — volume leveling and sample rate conversion.
    Uses ffmpeg for reliable cross-platform processing.
    """

    @staticmethod
    async def normalize_volume(
        input_path: Path,
        output_path: Path | None = None,
        target_lufs: float = -16.0,
    ) -> Path:
        """
        Normalize audio volume using EBU R128 loudness normalization.

        Args:
            input_path: Path to input audio.
            output_path: Path to output audio. If None, overwrites input.
            target_lufs: Target loudness in LUFS (default -16.0).

        Returns:
            Path to normalized audio file.
        """
        if not input_path.exists():
            raise AudioProcessingError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix(".normalized.wav")

        ffmpeg_path = _resolve_ffmpeg()
        if not ffmpeg_path:
            raise AudioProcessingError("ffmpeg not found in PATH")

        try:
            cmd = [
                ffmpeg_path,
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(output_path),
            ]

            def _run():
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if res.returncode != 0:
                    err = res.stderr.decode(errors="replace") if res.stderr else "Unknown error"
                    raise AudioProcessingError(f"Volume normalization failed: {err[:500]}")

            await asyncio.to_thread(_run)

            log.info(
                "volume_normalized",
                input=str(input_path),
                output=str(output_path),
                target_lufs=target_lufs,
            )

            return output_path

        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Volume normalization failed ({type(e).__name__}): {e}")

    @staticmethod
    async def convert_sample_rate(
        input_path: Path,
        output_path: Path | None = None,
        target_rate: int = 16000,
    ) -> Path:
        """
        Convert audio sample rate.

        Args:
            input_path: Path to input audio.
            output_path: Output path. If None, creates new file.
            target_rate: Target sample rate in Hz.

        Returns:
            Path to resampled audio file.
        """
        if output_path is None:
            output_path = input_path.with_suffix(f".{target_rate}hz.wav")

        ffmpeg_path = _resolve_ffmpeg()
        if not ffmpeg_path:
            raise AudioProcessingError("ffmpeg not found in PATH")

        try:
            cmd = [
                ffmpeg_path,
                "-i", str(input_path),
                "-ar", str(target_rate),
                "-ac", "1",
                "-y",
                str(output_path),
            ]

            def _run():
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if res.returncode != 0:
                    raise AudioProcessingError("Sample rate conversion failed")

            await asyncio.to_thread(_run)

            log.info(
                "sample_rate_converted",
                input=str(input_path),
                output=str(output_path),
                target_rate=target_rate,
            )

            return output_path

        except AudioProcessingError:
            raise
        except Exception as e:
            raise AudioProcessingError(f"Sample rate conversion failed ({type(e).__name__}): {e}")
