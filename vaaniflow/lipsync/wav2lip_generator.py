"""
Wav2Lip-based lip-sync video generation.

Takes original video + dubbed audio and generates a lip-synced video
where the speaker's lip movements match the dubbed audio track.

Wav2Lip is a neural network that synthesizes realistic lip movements
to match any target audio, given a video of a talking face.

Model: Wav2Lip or Wav2Lip-GAN (~170MB checkpoint)
  - Input: face video + target audio
  - Output: lip-synced video
  - Uses face detection (s3fd) for face localization

Architecture:
  Primary: Wav2Lip subprocess inference (needs wav2lip installed)
  Fallback: JSON manifest export (always available)

Installation:
  pip install wav2lip  # or clone from https://github.com/Rudrabha/Wav2Lip
  Download checkpoint: wav2lip_gan.pth
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Common locations where Wav2Lip inference script might be found
_INFERENCE_SCRIPT_CANDIDATES = [
    Path("wav2lip/inference.py"),
    Path("third_party/wav2lip/inference.py"),
    Path("/app/wav2lip/inference.py"),
    Path("/opt/wav2lip/inference.py"),
]

# Common locations for Wav2Lip checkpoint
_CHECKPOINT_CANDIDATES = [
    Path("wav2lip_gan.pth"),
    Path("checkpoints/wav2lip_gan.pth"),
    Path("third_party/wav2lip/checkpoints/wav2lip_gan.pth"),
    Path("/app/wav2lip/checkpoints/wav2lip_gan.pth"),
]


class Wav2LipGenerator:
    """
    Lip-sync video generator using Wav2Lip.

    Runs Wav2Lip inference as a subprocess to generate lip-synced
    video from an original video + dubbed audio track.

    Falls back gracefully if Wav2Lip is not installed — the pipeline
    continues without lip-sync (returns original video path).

    Usage:
        generator = Wav2LipGenerator(enabled=True)
        result_path = await generator.generate(
            original_video_path=Path("input.mp4"),
            dubbed_audio_path=Path("dubbed.wav"),
            output_path=Path("outputs/lipsync_job123.mp4"),
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        checkpoint_path: Optional[str] = None,
        timeout_seconds: int = 600,  # 10 minutes max for long videos
    ):
        self.enabled = enabled
        self.checkpoint_path = checkpoint_path
        self.timeout_seconds = timeout_seconds
        self._inference_script: Optional[Path] = None
        self._checkpoint: Optional[Path] = None
        self._wav2lip_available = self._check_wav2lip()

    def _check_wav2lip(self) -> bool:
        """Check if Wav2Lip inference script and checkpoint are available."""
        # Find inference script
        for candidate in _INFERENCE_SCRIPT_CANDIDATES:
            if candidate.exists():
                self._inference_script = candidate
                break

        # Find checkpoint
        if self.checkpoint_path:
            cp = Path(self.checkpoint_path)
            if cp.exists():
                self._checkpoint = cp
        else:
            for candidate in _CHECKPOINT_CANDIDATES:
                if candidate.exists():
                    self._checkpoint = candidate
                    break

        available = self._inference_script is not None and self._checkpoint is not None

        if not available:
            log.info(
                "wav2lip_not_available",
                inference_script=str(self._inference_script),
                checkpoint=str(self._checkpoint),
                message="Lip-sync generation will export JSON manifest only",
            )

        return available

    @property
    def is_available(self) -> bool:
        """Whether Wav2Lip can generate lip-synced video."""
        return self.enabled and self._wav2lip_available

    async def generate(
        self,
        original_video_path: Path,
        dubbed_audio_path: Path,
        output_path: Path,
    ) -> Optional[Path]:
        """
        Generate lip-synced video from original video + dubbed audio.

        Returns:
            Path to generated lip-synced video, or None if unavailable.
        """
        if not self.enabled:
            log.debug("wav2lip_disabled")
            return None

        if not self._wav2lip_available:
            log.info(
                "wav2lip_skipped",
                reason="Wav2Lip not installed or checkpoint missing",
            )
            return None

        if not original_video_path.exists():
            log.warning("wav2lip_input_missing", path=str(original_video_path))
            return None

        if not dubbed_audio_path.exists():
            log.warning("wav2lip_audio_missing", path=str(dubbed_audio_path))
            return None

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result = await asyncio.to_thread(
                self._run_inference,
                original_video_path,
                dubbed_audio_path,
                output_path,
            )

            if result and result.exists() and result.stat().st_size > 0:
                log.info(
                    "wav2lip_generated",
                    output=str(result),
                    size_mb=round(result.stat().st_size / 1024 / 1024, 1),
                )
                return result

            log.warning("wav2lip_output_empty", expected=str(output_path))
            return None

        except subprocess.TimeoutExpired:
            log.error(
                "wav2lip_timeout",
                timeout_seconds=self.timeout_seconds,
                video=str(original_video_path),
            )
            return None
        except Exception as e:
            log.error("wav2lip_generation_failed", error=str(e))
            return None

    def _run_inference(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Optional[Path]:
        """Run Wav2Lip inference subprocess. Runs in thread pool."""
        cmd = [
            "python",
            str(self._inference_script),
            "--checkpoint_path", str(self._checkpoint),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--pads", "0", "0", "0", "0",
            "--face_det_batch_size", "4",
            "--wav2lip_batch_size", "1",
            "--resize_factor", "1",
            "--nosmooth",
        ]

        log.info("wav2lip_inference_started", video=str(video_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )

        if result.returncode != 0:
            log.warning(
                "wav2lip_inference_error",
                returncode=result.returncode,
                stderr=result.stderr[-500:] if result.stderr else "",
            )
            return None

        return output_path
