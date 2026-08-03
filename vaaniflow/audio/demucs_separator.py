"""
Neural ambient separation using Demucs (Meta/Facebook Research).

Upgrades the spectral subtraction approach with state-of-the-art
neural source separation. Demucs separates audio into 4 stems:
vocals, drums, bass, other — allowing us to isolate background
music with far higher quality than frequency-domain heuristics.

Model: htdemucs (hybrid transformer, best quality/speed trade-off)
  - 4-stem separation: vocals, drums, bass, other
  - Ambient = drums + bass + other (everything except vocals)
  - ~80MB model, CPU-friendly (~10s for 1 minute of audio)

Architecture:
  Primary: Demucs neural source separation (SOTA quality)
  Fallback: scipy spectral subtraction (always available)
"""

import asyncio
import importlib.util
import io
from typing import Any

import structlog

from vaaniflow.audio.ambient_separator import (
    AmbientAudioPreserver as SpectralSeparator,
)
from vaaniflow.audio.ambient_separator import (
    SeparationResult,
)
from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


class DemucsAmbientPreserver:
    """
    Neural source separation with spectral subtraction fallback.

    Uses Meta's Demucs model for high-quality vocal/instrumental
    separation. Falls back to scipy spectral subtraction if Demucs
    or its dependencies (torch) are not installed.

    Drop-in replacement for AmbientAudioPreserver:
        preserver = DemucsAmbientPreserver(enabled=True)
        result = await preserver.separate(raw_audio_bytes)
        remixed = await preserver.remix(dubbed_bytes, result.ambient_bytes)
    """

    def __init__(
        self,
        enabled: bool = True,
        ambient_gain_db: float = -6.0,
        fallback_to_spectral: bool = True,
    ):
        self.enabled = enabled
        self.ambient_gain_db = ambient_gain_db
        self.fallback_to_spectral = fallback_to_spectral
        self._spectral_fallback = SpectralSeparator(
            enabled=enabled, ambient_gain_db=ambient_gain_db
        )
        self._demucs_available = self._check_demucs()
        self._separator: Any = None

    def _check_demucs(self) -> bool:
        """Check if demucs and torch are installed."""
        has_demucs = importlib.util.find_spec("demucs") is not None
        has_torch = importlib.util.find_spec("torch") is not None
        if not (has_demucs and has_torch):
            log.info(
                "demucs_not_available",
                message="Install demucs and torch for neural source separation",
                fallback="spectral_subtraction",
            )
        return has_demucs and has_torch

    async def separate(self, audio_bytes: bytes) -> SeparationResult:
        """
        Separate vocals from ambient/background audio.

        Primary: Demucs 4-stem neural separation
        Fallback: scipy spectral subtraction

        Returns SeparationResult with vocals_bytes, ambient_bytes,
        ambient_level_db, and has_significant_ambient flag.
        """
        if not self.enabled or not audio_bytes:
            return SeparationResult(
                vocals_bytes=audio_bytes,
                ambient_bytes=b"",
                ambient_level_db=-96.0,
                has_significant_ambient=False,
            )

        if not self._demucs_available:
            if self.fallback_to_spectral:
                return await self._spectral_fallback.separate(audio_bytes)
            return SeparationResult(
                vocals_bytes=audio_bytes,
                ambient_bytes=b"",
                ambient_level_db=-96.0,
                has_significant_ambient=False,
            )

        try:
            return await asyncio.to_thread(self._separate_demucs_sync, audio_bytes)
        except Exception as e:
            log.warning("demucs_separation_failed", error=str(e), fallback="spectral")
            if self.fallback_to_spectral:
                return await self._spectral_fallback.separate(audio_bytes)
            raise AudioProcessingError(f"Demucs separation failed: {e}") from e

    def _separate_demucs_sync(self, audio_bytes: bytes) -> SeparationResult:
        """Synchronous Demucs separation. Runs in thread pool."""
        import numpy as np
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from pydub import AudioSegment as PydubSeg

        # Convert input to standardized WAV (44100Hz stereo, Demucs default)
        audio = PydubSeg.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(2).set_frame_rate(44100).set_sample_width(2)

        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Load as torch tensor
        waveform, sr = torchaudio.load(wav_buffer)
        # waveform shape: (channels, samples)

        # Load Demucs model (cached after first load)
        if self._separator is None:
            self._separator = get_model("htdemucs")
            self._separator.eval()

        # Apply model — returns tensor of shape (sources, channels, samples)
        # Sources order: drums, bass, other, vocals
        with torch.no_grad():
            sources = apply_model(
                self._separator,
                waveform.unsqueeze(0),  # Add batch dimension
                device="cpu",
            )
            sources = sources.squeeze(0)  # Remove batch dimension

        # Source indices for htdemucs: drums=0, bass=1, other=2, vocals=3
        vocals = sources[3]  # (channels, samples)
        ambient = sources[0] + sources[1] + sources[2]  # drums + bass + other

        # Convert to bytes
        vocals_bytes = self._tensor_to_wav_bytes(vocals, sr)
        ambient_bytes = self._tensor_to_wav_bytes(ambient, sr)

        # Compute ambient level in dB
        ambient_np = ambient.numpy()
        ambient_rms = float(np.sqrt(np.mean(ambient_np**2)))
        ambient_db = float(20 * np.log10(max(ambient_rms, 1e-10)))
        has_significant = ambient_db > -40.0

        log.info(
            "demucs_separation_complete",
            ambient_level_db=round(ambient_db, 1),
            has_significant_ambient=has_significant,
            model="htdemucs",
        )

        return SeparationResult(
            vocals_bytes=vocals_bytes,
            ambient_bytes=ambient_bytes,
            ambient_level_db=ambient_db,
            has_significant_ambient=has_significant,
        )

    @staticmethod
    def _tensor_to_wav_bytes(tensor, sample_rate: int) -> bytes:
        """Convert a torch tensor (channels, samples) to WAV bytes."""
        buf = io.BytesIO()
        import torchaudio

        torchaudio.save(buf, tensor, sample_rate, format="wav")
        return buf.getvalue()

    async def remix(self, dubbed_audio_bytes: bytes, ambient_bytes: bytes) -> bytes:
        """
        Layer dubbed speech with original ambient audio.

        Delegates to the spectral fallback's pydub-based mixer,
        which works identically regardless of how the ambient
        was originally separated.
        """
        if not ambient_bytes or not self.enabled:
            return dubbed_audio_bytes

        # Remix logic is the same regardless of separation method
        return await self._spectral_fallback.remix(dubbed_audio_bytes, ambient_bytes)
