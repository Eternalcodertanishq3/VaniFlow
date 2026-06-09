"""
AmbientAudioPreserver — separate background audio from speech and re-layer after dubbing.

This is a unique original feature of VaaniFlow.
Most dubbing pipelines discard background music/ambient. We preserve it.

Technique: Spectral subtraction using scipy.
  1. Compute STFT of mixed audio
  2. Estimate noise/ambient profile from non-speech regions
  3. Subtract speech spectrum to isolate ambient
  4. After dubbing: layer dubbed speech + original ambient
"""
import asyncio
import io
import structlog
from dataclasses import dataclass
from typing import Optional

log = structlog.get_logger(__name__)


@dataclass
class SeparationResult:
    vocals_bytes: bytes
    ambient_bytes: bytes
    ambient_level_db: float
    has_significant_ambient: bool


class AmbientAudioPreserver:
    """
    Separates ambient/background audio from speech.
    Re-layers it after dubbing for a natural sound.
    """

    def __init__(self, enabled: bool = True, ambient_gain_db: float = -6.0):
        self.enabled = enabled
        self.ambient_gain_db = ambient_gain_db
        self._scipy_available = None

    def _check_scipy(self) -> bool:
        if self._scipy_available is None:
            try:
                import scipy
                import numpy
                self._scipy_available = True
            except ImportError:
                log.warning(
                    "scipy_not_installed",
                    message="pip install scipy numpy for ambient separation",
                )
                self._scipy_available = False
        return self._scipy_available

    async def separate(self, audio_bytes: bytes) -> SeparationResult:
        """
        Separate vocals from ambient audio.
        Returns both streams as bytes.
        """
        if not self.enabled or not self._check_scipy():
            return SeparationResult(
                vocals_bytes=audio_bytes,
                ambient_bytes=b"",
                ambient_level_db=-96.0,
                has_significant_ambient=False,
            )

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._separate_sync, audio_bytes)
        except Exception as e:
            log.warning("ambient_separation_failed", error=str(e))
            return SeparationResult(
                vocals_bytes=audio_bytes,
                ambient_bytes=b"",
                ambient_level_db=-96.0,
                has_significant_ambient=False,
            )

    def _separate_sync(self, audio_bytes: bytes) -> SeparationResult:
        """Synchronous spectral subtraction."""
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import stft, istft

        try:
            from pydub import AudioSegment as PydubSeg

            # Convert to WAV format for scipy processing
            audio = PydubSeg.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            sr, data = wavfile.read(wav_buffer)
            data = data.astype(np.float32) / 32768.0

            # STFT
            nperseg = 1024
            f, t, Zxx = stft(data, fs=sr, nperseg=nperseg)

            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            # Estimate ambient: use the bottom 20th percentile of magnitude as noise floor
            noise_estimate = np.percentile(magnitude, 20, axis=1, keepdims=True)

            # Spectral subtraction
            vocals_mag = np.maximum(magnitude - noise_estimate * 1.5, 0)
            ambient_mag = np.minimum(magnitude, noise_estimate * 2.0)

            # Reconstruct
            vocals_complex = vocals_mag * np.exp(1j * phase)
            ambient_complex = ambient_mag * np.exp(1j * phase)

            _, vocals_signal = istft(vocals_complex, fs=sr, nperseg=nperseg)
            _, ambient_signal = istft(ambient_complex, fs=sr, nperseg=nperseg)

            # Compute ambient level
            ambient_rms = float(np.sqrt(np.mean(ambient_signal ** 2)))
            ambient_db = 20 * np.log10(max(ambient_rms, 1e-10))
            has_significant = ambient_db > -40

            # Convert back to bytes
            vocals_bytes = self._signal_to_bytes(vocals_signal, sr)
            ambient_bytes_out = self._signal_to_bytes(ambient_signal, sr)

            log.info(
                "ambient_separated",
                ambient_db=round(ambient_db, 1),
                has_significant_ambient=has_significant,
            )

            return SeparationResult(
                vocals_bytes=vocals_bytes,
                ambient_bytes=ambient_bytes_out,
                ambient_level_db=ambient_db,
                has_significant_ambient=has_significant,
            )

        except Exception as e:
            log.warning("spectral_subtraction_failed", error=str(e))
            return SeparationResult(
                vocals_bytes=audio_bytes,
                ambient_bytes=b"",
                ambient_level_db=-96.0,
                has_significant_ambient=False,
            )

    async def remix(self, dubbed_audio_bytes: bytes, ambient_bytes: bytes) -> bytes:
        """
        Layer dubbed speech with original ambient audio.
        """
        if not ambient_bytes or not self.enabled:
            return dubbed_audio_bytes

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._remix_sync, dubbed_audio_bytes, ambient_bytes
            )
        except Exception as e:
            log.warning("ambient_remix_failed", error=str(e))
            return dubbed_audio_bytes

    def _remix_sync(self, dubbed_bytes: bytes, ambient_bytes: bytes) -> bytes:
        """Synchronous remix using pydub."""
        from pydub import AudioSegment as PydubSeg

        dubbed = PydubSeg.from_file(io.BytesIO(dubbed_bytes))
        ambient = PydubSeg.from_file(io.BytesIO(ambient_bytes))

        # Match lengths
        if len(ambient) > len(dubbed):
            ambient = ambient[:len(dubbed)]
        elif len(ambient) < len(dubbed):
            silence = PydubSeg.silent(duration=len(dubbed) - len(ambient))
            ambient = ambient + silence

        # Apply gain to ambient and overlay
        ambient = ambient + self.ambient_gain_db
        mixed = dubbed.overlay(ambient)

        output_buffer = io.BytesIO()
        mixed.export(output_buffer, format="wav")
        return output_buffer.getvalue()

    def _signal_to_bytes(self, signal, sr: int) -> bytes:
        """Convert numpy signal to WAV bytes."""
        import numpy as np
        from scipy.io import wavfile

        signal = np.clip(signal, -1.0, 1.0)
        signal_int16 = (signal * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sr, signal_int16)
        return buffer.getvalue()
