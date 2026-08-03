"""
Video Lip-Sync Pipeline Step.

This module provides lip synchronization capabilities for VaaniFlow.

Primary: MuseTalk neural lip-sync generation (if installed)
Fallback: JSON timing manifest export (always available)

Pipeline integration point:
  After audio stitching (Stage 6), before final output delivery.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import structlog

from vaaniflow.lipsync.musetalk_generator import MuseTalkGenerator
from vaaniflow.models import AudioSegment

log = structlog.get_logger(__name__)


@dataclass
class LipSyncSegment:
    """A single segment's timing and metadata for lip-sync alignment."""

    index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    original_text: str
    translated_text: str
    emotion_label: Optional[str] = None
    speaking_rate: Optional[float] = None


@dataclass
class LipSyncManifest:
    """
    Complete lip-sync alignment manifest.

    This JSON-serializable manifest contains all the information
    a downstream video renderer needs to align lip movements
    with dubbed audio segments.
    """

    job_id: str
    source_language: str
    target_language: str
    total_duration_ms: float
    dubbed_audio_path: str
    original_video_path: Optional[str] = None
    segments: Optional[list[LipSyncSegment]] = None
    renderer: str = "musetalk"

    def __post_init__(self):
        if self.segments is None:
            self.segments = []

    def to_dict(self) -> dict:
        return asdict(self)


class LipSyncExporter:
    """
    Exports lip-sync data — either as a MuseTalk-generated video
    or a JSON timing manifest for downstream renderers.

    Primary: MuseTalk neural lip-sync video generation
    Fallback: JSON manifest with segment timestamps

    Usage in pipeline:
        exporter = LipSyncExporter(enabled=True)
        result = await exporter.export(
            segments=tts_result.segments,
            job_id=job.job_id,
            dubbed_audio_path=output_path,
            source_language="en",
            target_language="hi",
            total_duration_ms=transcription.total_duration_ms,
            original_video_path=input_path,
        )
    """

    def __init__(self, enabled: bool = False, output_dir: str = "outputs"):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.musetalk = MuseTalkGenerator(enabled=enabled)

    async def export(
        self,
        segments: list[AudioSegment],
        job_id: str,
        dubbed_audio_path: Path,
        source_language: str,
        target_language: str,
        total_duration_ms: float,
        original_video_path: Optional[Path] = None,
        emotions: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Export lip-sync output.

        1. If MuseTalk is available and input is a video:
           → Generate lip-synced video
        2. Always export JSON manifest as well
           → For downstream renderers or manual inspection

        Returns:
            Path to generated lip-synced video or JSON manifest.
        """
        if not self.enabled:
            log.debug("lipsync_export_disabled")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build manifest segments
        sync_segments = self._build_sync_segments(segments, emotions)

        # Build manifest
        manifest = LipSyncManifest(
            job_id=job_id,
            source_language=source_language,
            target_language=target_language,
            total_duration_ms=total_duration_ms,
            dubbed_audio_path=str(dubbed_audio_path),
            original_video_path=str(original_video_path) if original_video_path else None,
            segments=sync_segments,
        )

        # Always export JSON manifest
        manifest_path = await self._export_manifest(manifest)

        # Try MuseTalk if available and input is video
        if (
            self.musetalk.is_available
            and original_video_path
            and original_video_path.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi", ".mov"}
        ):
            lipsync_output = self.output_dir / f"{job_id}_lipsync.mp4"
            result = await self.musetalk.generate(
                original_video_path=original_video_path,
                dubbed_audio_path=dubbed_audio_path,
                output_path=lipsync_output,
            )
            if result:
                log.info(
                    "lipsync_video_generated",
                    job_id=job_id,
                    path=str(result),
                )
                return result

        log.info(
            "lipsync_manifest_exported",
            job_id=job_id,
            segments=len(sync_segments),
            path=str(manifest_path),
        )
        return manifest_path

    def _build_sync_segments(
        self,
        segments: list[AudioSegment],
        emotions: Optional[dict] = None,
    ) -> list[LipSyncSegment]:
        """Build LipSyncSegment list from pipeline AudioSegments."""
        sync_segments = []
        for seg in segments:
            emotion_label = None
            speaking_rate = None
            if emotions and seg.index in emotions:
                emo = emotions[seg.index]
                emotion_label = emo.label.value if hasattr(emo, "label") else str(emo)
                speaking_rate = emo.speaking_rate if hasattr(emo, "speaking_rate") else None

            sync_segments.append(
                LipSyncSegment(
                    index=seg.index,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    duration_ms=seg.duration_ms,
                    original_text=seg.original_text,
                    translated_text=seg.translated_text or "",
                    emotion_label=emotion_label,
                    speaking_rate=speaking_rate,
                )
            )
        return sync_segments

    async def _export_manifest(self, manifest: LipSyncManifest) -> Path:
        """Write manifest JSON to disk."""
        manifest_path = self.output_dir / f"{manifest.job_id}_lipsync_manifest.json"
        await asyncio.to_thread(self._write_manifest, manifest_path, manifest)
        return manifest_path

    @staticmethod
    def _write_manifest(path: Path, manifest: LipSyncManifest):
        """Write manifest to disk (runs in thread pool)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
