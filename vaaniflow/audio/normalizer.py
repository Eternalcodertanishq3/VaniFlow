"""
Audio normalization utilities.
Volume leveling and sample rate conversion.
"""
import asyncio
from pathlib import Path
import structlog

from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


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

        try:
            # Two-pass loudness normalization using ffmpeg loudnorm filter
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                "-ar", "16000",
                "-ac", "1",
                "-y",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise AudioProcessingError(
                    f"Volume normalization failed: {error_msg}"
                )

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
            raise AudioProcessingError(f"Volume normalization failed: {e}")

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

        try:
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-ar", str(target_rate),
                "-ac", "1",
                "-y",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            if process.returncode != 0:
                raise AudioProcessingError("Sample rate conversion failed")

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
            raise AudioProcessingError(f"Sample rate conversion failed: {e}")
