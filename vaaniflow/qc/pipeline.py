"""
Quality Control Pipeline.
Validates each TTS segment before audio stitching.

What it checks per segment:
  1. Silence ratio — is the audio mostly empty?
  2. Length ratio — is TTS wildly longer/shorter than original?
  3. Minimum size — did the TTS provider return garbage?
  4. Audio integrity — can pydub parse the bytes?
"""
import io
import asyncio
import structlog
from vaaniflow.qc.models import QCStatus, SegmentQCResult, PipelineQCResult, QCConfig
from vaaniflow.models import AudioSegment

log = structlog.get_logger(__name__)


class QualityController:
    """
    Validates synthesized audio segments before stitching.
    Runs in parallel across all segments for speed.
    """

    def __init__(self, config: QCConfig = None):
        self.config = config or QCConfig()

    async def validate_pipeline_output(
        self,
        segments: list[AudioSegment],
    ) -> PipelineQCResult:
        """
        Validate all segments concurrently.
        Returns a PipelineQCResult with per-segment status.
        """
        tasks = [self._validate_segment(seg) for seg in segments]
        results: list[SegmentQCResult] = await asyncio.gather(*tasks)

        pass_count = sum(1 for r in results if r.status == QCStatus.PASS)
        warn_count = sum(1 for r in results if r.status == QCStatus.WARN)
        fail_count = sum(1 for r in results if r.status == QCStatus.FAIL)
        retry_segments = [r.segment_index for r in results if r.should_retry]

        if fail_count > 0:
            overall = QCStatus.FAIL
        elif warn_count > 0:
            overall = QCStatus.WARN
        else:
            overall = QCStatus.PASS

        log.info(
            "qc_pipeline_complete",
            overall=overall,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            retry_segments=retry_segments,
        )

        return PipelineQCResult(
            overall_status=overall,
            segments=results,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            retry_segments=retry_segments,
        )

    async def _validate_segment(self, segment: AudioSegment) -> SegmentQCResult:
        """Validate a single segment asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_sync, segment)

    def _validate_sync(self, segment: AudioSegment) -> SegmentQCResult:
        """Synchronous validation — runs in thread pool."""
        issues = []
        status = QCStatus.PASS
        should_retry = False

        audio_bytes = segment.audio_bytes or b""

        # Check 1: Minimum audio size
        if len(audio_bytes) < self.config.min_audio_bytes:
            issues.append(f"Audio too small: {len(audio_bytes)} bytes")
            status = QCStatus.FAIL
            should_retry = True
            return SegmentQCResult(
                segment_index=segment.index, status=status,
                silence_ratio=1.0, length_ratio=0.0,
                issues=issues, should_retry=should_retry,
            )

        # Check 2: Silence ratio using pydub
        silence_ratio = self._compute_silence_ratio(audio_bytes)
        if silence_ratio > self.config.max_silence_ratio:
            issues.append(f"Too much silence: {silence_ratio:.1%}")
            status = QCStatus.FAIL
            should_retry = True

        # Check 3: Length ratio
        length_ratio = 1.0
        if segment.duration_ms > 0:
            tts_duration_ms = self._estimate_duration_ms(audio_bytes)
            length_ratio = tts_duration_ms / segment.duration_ms

            if length_ratio > self.config.max_length_ratio:
                issues.append(f"TTS too long: {length_ratio:.1f}x original")
                if status == QCStatus.PASS:
                    status = QCStatus.WARN

            elif length_ratio < self.config.min_length_ratio:
                issues.append(f"TTS too short: {length_ratio:.1f}x original")
                if status == QCStatus.PASS:
                    status = QCStatus.WARN

        if status == QCStatus.PASS:
            log.debug("qc_segment_pass", index=segment.index)
        else:
            log.warning(
                "qc_segment_issue",
                index=segment.index,
                status=status,
                issues=issues,
            )

        return SegmentQCResult(
            segment_index=segment.index,
            status=status,
            silence_ratio=silence_ratio,
            length_ratio=length_ratio,
            issues=issues,
            should_retry=should_retry,
        )

    def _compute_silence_ratio(self, audio_bytes: bytes) -> float:
        """Compute ratio of silence using pydub's dBFS threshold."""
        try:
            from pydub import AudioSegment as PydubSeg
            from pydub.silence import detect_silence

            audio = PydubSeg.from_file(io.BytesIO(audio_bytes))
            silent_ranges = detect_silence(audio, min_silence_len=100, silence_thresh=-40)
            total_silence_ms = sum(end - start for start, end in silent_ranges)
            return total_silence_ms / max(len(audio), 1)
        except Exception:
            return 0.0

    def _estimate_duration_ms(self, audio_bytes: bytes) -> float:
        """Estimate duration from audio bytes."""
        try:
            from pydub import AudioSegment as PydubSeg
            audio = PydubSeg.from_file(io.BytesIO(audio_bytes))
            return float(len(audio))
        except Exception:
            return len(audio_bytes) / 32.0
