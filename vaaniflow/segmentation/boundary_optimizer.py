"""
SmartSegmentBoundaryOptimizer — fix Whisper's raw segment cuts.

Problem: Whisper segments on silence, not sentence boundaries.
  Segment 1: "The quick brown fox"       (end_ms: 2000)
  Segment 2: "jumped over the lazy dog"  (start_ms: 2050)

These should be one segment for translation quality.
Splitting them causes unnatural TTS and awkward dubbing.

Solution:
  1. Run spaCy sentence tokenizer on all transcription text
  2. Detect which Whisper segments break natural sentence boundaries
  3. Merge adjacent segments that form complete sentences
  4. Preserve original timing (merged segment spans both time ranges)
"""
import asyncio
import structlog
from vaaniflow.models import AudioSegment, TranscriptionResult

log = structlog.get_logger(__name__)

MAX_MERGE_GAP_MS = 800   # Only merge segments within 800ms of each other
MAX_MERGED_WORDS = 50    # Don't create segments longer than 50 words


class SmartSegmentBoundaryOptimizer:
    """
    Merges incomplete Whisper segments into complete sentences
    using spaCy's sentence boundary detection.
    """

    def __init__(self, enabled: bool = True, language: str = "en"):
        self.enabled = enabled
        self.language = language
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    import subprocess
                    subprocess.run(
                        ["python", "-m", "spacy", "download", "en_core_web_sm"],
                        capture_output=True
                    )
                    self._nlp = spacy.load("en_core_web_sm")
            except ImportError:
                log.warning("spacy_not_installed", message="pip install spacy for boundary optimization")
                return None
        return self._nlp

    async def optimize(
        self, transcription: TranscriptionResult
    ) -> TranscriptionResult:
        """
        Optimize segment boundaries.
        Returns same TranscriptionResult with merged segments where appropriate.
        """
        if not self.enabled or len(transcription.segments) <= 1:
            return transcription

        nlp = self._get_nlp()
        if nlp is None:
            return transcription

        try:
            loop = asyncio.get_event_loop()
            optimized_segments = await loop.run_in_executor(
                None, self._optimize_sync, transcription.segments, nlp
            )

            log.info(
                "segment_boundaries_optimized",
                original_count=len(transcription.segments),
                optimized_count=len(optimized_segments),
                merged=len(transcription.segments) - len(optimized_segments),
            )

            return TranscriptionResult(
                segments=optimized_segments,
                source_language=transcription.source_language,
                total_duration_ms=transcription.total_duration_ms,
                provider_used=transcription.provider_used,
            )
        except Exception as e:
            log.warning("boundary_optimization_failed", error=str(e))
            return transcription

    def _optimize_sync(self, segments: list[AudioSegment], nlp) -> list[AudioSegment]:
        """Synchronous boundary optimization."""
        if not segments:
            return segments

        full_text = " ".join(seg.original_text.strip() for seg in segments)
        doc = nlp(full_text)
        sentence_boundaries = {
            sent.end_char for sent in doc.sents
        }

        merged = []
        buffer_segments = [segments[0]]
        char_pos = len(segments[0].original_text)

        for seg in segments[1:]:
            prev = buffer_segments[-1]
            gap_ms = seg.start_ms - prev.end_ms

            at_sentence_end = char_pos in sentence_boundaries
            too_long = sum(
                len(s.original_text.split()) for s in buffer_segments
            ) >= MAX_MERGED_WORDS
            gap_too_large = gap_ms > MAX_MERGE_GAP_MS

            if at_sentence_end or too_long or gap_too_large:
                merged.append(self._merge_segments(buffer_segments, len(merged)))
                buffer_segments = [seg]
            else:
                buffer_segments.append(seg)

            char_pos += 1 + len(seg.original_text)  # +1 for space

        if buffer_segments:
            merged.append(self._merge_segments(buffer_segments, len(merged)))

        return merged

    def _merge_segments(
        self, segments: list[AudioSegment], new_index: int
    ) -> AudioSegment:
        """Merge multiple segments into one, preserving timing."""
        merged_text = " ".join(s.original_text.strip() for s in segments)
        return AudioSegment(
            index=new_index,
            start_ms=segments[0].start_ms,
            end_ms=segments[-1].end_ms,
            duration_ms=segments[-1].end_ms - segments[0].start_ms,
            original_text=merged_text,
        )
