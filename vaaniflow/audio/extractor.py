"""
Extract audio track from video/audio files.
Supports ffmpeg (preferred) and pydub (fallback) for format conversion.
WAV files are passed through with minimal processing when ffmpeg is unavailable.
"""
import asyncio
import shutil
import tempfile
import wave
from pathlib import Path
import structlog

from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)

# Target format for downstream processing (Whisper expects 16kHz mono)
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def _resolve_ffmpeg() -> str | None:
    """Resolve ffmpeg to its full path. Returns None if not found."""
    return shutil.which("ffmpeg")


def _resolve_ffprobe() -> str | None:
    """Resolve ffprobe to its full path. Returns None if not found."""
    return shutil.which("ffprobe")


class AudioExtractor:
    """
    Extract audio from video/audio files.
    Uses ffmpeg when available; falls back to pydub for audio-only files.
    Outputs normalized WAV for downstream processing.
    """

    @property
    def _has_ffmpeg(self) -> bool:
        """Check ffmpeg availability dynamically (not cached at init)."""
        return _resolve_ffmpeg() is not None

    async def extract(self, input_path: Path) -> Path:
        """
        Extract audio from a video/audio file.

        Strategy:
        1. If ffmpeg is available → use it (handles all formats)
        2. If input is .wav → copy/resample with pure Python
        3. Otherwise → try pydub (needs ffmpeg for non-wav, but handles more gracefully)

        Args:
            input_path: Path to input video/audio file.

        Returns:
            Path to extracted WAV audio file (16kHz mono).

        Raises:
            AudioProcessingError on failure.
        """
        if not input_path.exists():
            raise AudioProcessingError(f"Input file not found: {input_path}")

        suffix = input_path.suffix.lower()

        log.info(
            "audio_extraction_started",
            input=str(input_path),
            format=suffix,
            ffmpeg_available=self._has_ffmpeg,
        )

        # Strategy 1: Use ffmpeg if available (handles everything)
        if self._has_ffmpeg:
            return await self._extract_with_ffmpeg(input_path)

        # Strategy 2: WAV files — use pure Python wave module
        if suffix == ".wav":
            return await self._passthrough_wav(input_path)

        # Strategy 3: Non-WAV audio files — try pydub
        if suffix in {".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"}:
            return await self._extract_with_pydub(input_path)

        # Strategy 4: Video files absolutely need ffmpeg
        if suffix in {".mp4", ".mkv", ".avi", ".mov"}:
            raise AudioProcessingError(
                f"Video files ({suffix}) require ffmpeg for audio extraction. "
                f"Install ffmpeg: winget install ffmpeg  OR  "
                f"choco install ffmpeg  OR  download from https://ffmpeg.org/download.html "
                f"and add to PATH."
            )

        raise AudioProcessingError(
            f"Unsupported format '{suffix}' and ffmpeg is not installed. "
            f"Install ffmpeg for full format support."
        )

    async def _extract_with_ffmpeg(self, input_path: Path) -> Path:
        """Extract audio using ffmpeg subprocess (preferred method)."""
        output_path = Path(tempfile.mktemp(suffix=".wav"))

        def _run_ffmpeg():
            import subprocess
            ffmpeg_path = _resolve_ffmpeg()
            if not ffmpeg_path:
                raise AudioProcessingError("ffmpeg not found in PATH")
            cmd = [
                ffmpeg_path,              # use full resolved path (Windows compat)
                "-i", str(input_path),
                "-vn",                    # no video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", str(TARGET_CHANNELS),
                "-y",                     # overwrite
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode(errors="replace") if result.stderr else "Unknown ffmpeg error"
                raise AudioProcessingError(
                    f"ffmpeg extraction failed (code {result.returncode}): {error_msg[:500]}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AudioProcessingError("ffmpeg produced empty output file")

            return output_path

        try:
            output_path = await asyncio.to_thread(_run_ffmpeg)

            log.info(
                "audio_extraction_completed",
                method="ffmpeg",
                output=str(output_path),
                size_bytes=output_path.stat().st_size,
            )
            return output_path

        except AudioProcessingError:
            raise
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffmpeg not found. Install: winget install ffmpeg"
            )
        except Exception as e:
            raise AudioProcessingError(f"Audio extraction failed ({type(e).__name__}): {e}")

    async def _extract_with_pydub(self, input_path: Path) -> Path:
        """
        Extract audio using pydub (requires ffmpeg for MP3/OGG decoding).
        This provides better error messages than raw subprocess.
        """
        def _convert():
            try:
                from pydub import AudioSegment as PydubSegment
            except ImportError:
                raise AudioProcessingError(
                    "pydub is not installed. Run: pip install pydub"
                )

            try:
                audio = PydubSegment.from_file(str(input_path))

                # Convert to 16kHz mono 16-bit
                audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
                audio = audio.set_channels(TARGET_CHANNELS)
                audio = audio.set_sample_width(2)  # 16-bit

                output_path = Path(tempfile.mktemp(suffix=".wav"))
                audio.export(str(output_path), format="wav")

                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise AudioProcessingError("pydub produced empty output")

                return output_path

            except AudioProcessingError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                if "ffmpeg" in error_str or "ffprobe" in error_str or "errno" in error_str:
                    raise AudioProcessingError(
                        f"Cannot decode '{input_path.suffix}' without ffmpeg. "
                        f"Install ffmpeg: winget install ffmpeg  OR  "
                        f"choco install ffmpeg  OR  download from "
                        f"https://ffmpeg.org/download.html and add to PATH."
                    )
                raise AudioProcessingError(f"pydub conversion failed: {e}")

        output_path = await asyncio.to_thread(_convert)
        log.info(
            "audio_extraction_completed",
            method="pydub",
            output=str(output_path),
            size_bytes=output_path.stat().st_size,
        )
        return output_path

    async def _passthrough_wav(self, input_path: Path) -> Path:
        """
        Handle WAV files without ffmpeg.
        Checks if already 16kHz mono — if so, just copies.
        Otherwise resamples with pure Python (basic quality).
        """
        def _process_wav():
            try:
                with wave.open(str(input_path), 'rb') as wav_in:
                    framerate = wav_in.getframerate()
                    channels = wav_in.getnchannels()
                    sampwidth = wav_in.getsampwidth()
                    frames = wav_in.readframes(wav_in.getnframes())

                # If already correct format, just copy
                if framerate == TARGET_SAMPLE_RATE and channels == TARGET_CHANNELS:
                    output_path = Path(tempfile.mktemp(suffix=".wav"))
                    import shutil
                    shutil.copy2(str(input_path), str(output_path))
                    return output_path

                # Basic resampling: convert to target format
                # For production quality, ffmpeg is strongly recommended
                import struct

                # Convert to mono if stereo
                if channels == 2 and sampwidth == 2:
                    samples = struct.unpack(f'<{len(frames) // 2}h', frames)
                    mono_samples = [
                        (samples[i] + samples[i + 1]) // 2
                        for i in range(0, len(samples), 2)
                    ]
                    frames = struct.pack(f'<{len(mono_samples)}h', *mono_samples)
                    channels = 1

                # Simple sample rate conversion (nearest-neighbor)
                if framerate != TARGET_SAMPLE_RATE and sampwidth == 2:
                    samples = struct.unpack(f'<{len(frames) // 2}h', frames)
                    ratio = TARGET_SAMPLE_RATE / framerate
                    new_length = int(len(samples) * ratio)
                    resampled = [
                        samples[min(int(i / ratio), len(samples) - 1)]
                        for i in range(new_length)
                    ]
                    frames = struct.pack(f'<{len(resampled)}h', *resampled)
                    framerate = TARGET_SAMPLE_RATE

                output_path = Path(tempfile.mktemp(suffix=".wav"))
                with wave.open(str(output_path), 'wb') as wav_out:
                    wav_out.setnchannels(TARGET_CHANNELS)
                    wav_out.setsampwidth(2)  # 16-bit
                    wav_out.setframerate(TARGET_SAMPLE_RATE)
                    wav_out.writeframes(frames)

                return output_path

            except Exception as e:
                raise AudioProcessingError(f"WAV processing failed: {e}")

        output_path = await asyncio.to_thread(_process_wav)
        log.info(
            "audio_extraction_completed",
            method="wave_passthrough",
            output=str(output_path),
            size_bytes=output_path.stat().st_size,
        )
        return output_path

    async def get_duration_ms(self, audio_path: Path) -> float:
        """Get duration of an audio file in milliseconds."""
        suffix = audio_path.suffix.lower()

        # For WAV files, use pure Python — no ffprobe needed
        if suffix == ".wav":
            return await self._get_wav_duration_ms(audio_path)

        # For other formats, try ffprobe
        if _resolve_ffprobe():
            return await self._get_duration_ffprobe(audio_path)

        # Last resort: try pydub
        return await self._get_duration_pydub(audio_path)

    async def _get_wav_duration_ms(self, audio_path: Path) -> float:
        """Get WAV duration using pure Python wave module."""
        def _read_duration():
            try:
                with wave.open(str(audio_path), 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    return (frames / rate) * 1000
            except Exception as e:
                raise AudioProcessingError(f"Failed to get WAV duration: {e}")

        return await asyncio.to_thread(_read_duration)

    async def _get_duration_ffprobe(self, audio_path: Path) -> float:
        """Get duration using ffprobe."""
        def _run():
            import subprocess
            ffprobe_path = _resolve_ffprobe()
            if not ffprobe_path:
                raise AudioProcessingError("ffprobe not found in PATH")
            cmd = [
                ffprobe_path,             # use full resolved path (Windows compat)
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )

            if result.returncode != 0:
                raise AudioProcessingError("ffprobe failed")

            duration_s = float(result.stdout.decode().strip())
            return duration_s * 1000

        try:
            return await asyncio.to_thread(_run)
        except (ValueError, AudioProcessingError) as e:
            raise AudioProcessingError(f"Failed to get audio duration: {e}")

    async def _get_duration_pydub(self, audio_path: Path) -> float:
        """Get duration using pydub as fallback."""
        def _read():
            try:
                from pydub import AudioSegment as PydubSegment
                audio = PydubSegment.from_file(str(audio_path))
                return len(audio)  # pydub returns duration in ms
            except Exception as e:
                raise AudioProcessingError(f"Failed to get duration: {e}")

        return await asyncio.to_thread(_read)
