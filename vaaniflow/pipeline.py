"""
VaaniFlow Pipeline Orchestrator — Phase 2.
Orchestrates all providers + Phase 2 features in the correct sequence.

Pipeline stages:
  1. Extract audio from video
  2. Ambient separation (if enabled)
  3. Transcribe speech to text
  4. Boundary optimization (if enabled)
  5. Translate (batch) + back-translation scoring
  6. Synthesize TTS (with emotion + pronunciation correction)
  7. QC validation (if enabled)
  8. Stitch audio
  9. Ambient remix (if enabled)
"""
import asyncio
import time
import structlog
from pathlib import Path
from structlog.contextvars import bind_contextvars, clear_contextvars

from vaaniflow.models import (
    DubbingJob, DubbingJobConfig, JobStatus,
    TranscriptionResult, TranslationResult, TTSResult,
    AudioSegment, TTSProvider, TranslationProvider,
)
from vaaniflow.providers.tts.elevenlabs_provider import ElevenLabsProvider
from vaaniflow.providers.tts.sarvam_provider import SarvamTTSProvider
from vaaniflow.providers.tts.gtts_provider import GTTSProvider
from vaaniflow.providers.translation.google_provider import GoogleTranslationProvider
from vaaniflow.providers.translation.sarvam_provider import SarvamTranslationProvider
from vaaniflow.providers.transcription.whisper_provider import WhisperProvider
from vaaniflow.audio.extractor import AudioExtractor
from vaaniflow.audio.stitcher import AudioStitcher
from vaaniflow.cache.redis_cache import TranslationCache
from vaaniflow.exceptions import PipelineError
from vaaniflow.config import settings

# Phase 2 imports
from vaaniflow.emotion.detector import EmotionPreserver
from vaaniflow.quality.back_translation import BackTranslationQualityScorer
from vaaniflow.segmentation.boundary_optimizer import SmartSegmentBoundaryOptimizer
from vaaniflow.pronunciation.corrector import IndianNamePronunciationCorrector
from vaaniflow.audio.ambient_separator import AmbientAudioPreserver
from vaaniflow.qc.pipeline import QualityController
from vaaniflow.qc.models import QCConfig, QCStatus
from vaaniflow.metrics import (
    JOBS_TOTAL, ACTIVE_JOBS, PIPELINE_STAGE_DURATION,
    TRANSLATION_CACHE_HITS, TRANSLATION_CACHE_MISSES,
    PROVIDER_ERRORS, TTS_AUDIO_BYTES, QC_SEGMENT_FAILURES,
    EMOTION_DETECTIONS, BACK_TRANSLATION_SCORES, BACK_TRANSLATION_RETRIES,
)

log = structlog.get_logger(__name__)


class VaaniFlowPipeline:
    """
    Main dubbing pipeline orchestrator — Phase 2.
    Manages the full flow with emotion, QC, boundary optimization,
    pronunciation correction, ambient preservation, and back-translation.
    """

    def __init__(self):
        self.extractor = AudioExtractor()
        self.stitcher = AudioStitcher()
        self.cache = TranslationCache()

        # Provider registry — pipeline selects based on config
        self.tts_providers = {
            TTSProvider.ELEVENLABS: ElevenLabsProvider(),
            TTSProvider.SARVAM: SarvamTTSProvider(),
            TTSProvider.GTTS: GTTSProvider(),  # always available fallback
        }
        self.translation_providers = {
            TranslationProvider.GOOGLE: GoogleTranslationProvider(),
            TranslationProvider.SARVAM: SarvamTranslationProvider(),
        }
        self.transcription_provider = WhisperProvider()

        # Phase 2: Feature modules
        self.emotion_preserver = EmotionPreserver(
            enabled=settings.emotion_detection_enabled
        )
        self.back_translation_scorer = BackTranslationQualityScorer(
            threshold=settings.back_translation_threshold,
            enabled=settings.back_translation_enabled,
        )
        self.boundary_optimizer = SmartSegmentBoundaryOptimizer(
            enabled=settings.boundary_optimization_enabled
        )
        self.pronunciation_corrector = IndianNamePronunciationCorrector(
            enabled=settings.pronunciation_correction_enabled
        )
        self.ambient_preserver = AmbientAudioPreserver(
            enabled=settings.ambient_separation_enabled
        )
        self.qc_controller = QualityController(
            config=QCConfig(
                max_silence_ratio=settings.qc_max_silence_ratio,
                max_length_ratio=settings.qc_max_length_ratio,
            )
        )

    async def run(self, job: DubbingJob, input_path: Path) -> Path:
        """
        Execute the full dubbing pipeline for a job.
        Each stage updates job.status and logs progress.
        """
        # Bind job context to all logs in this coroutine
        bind_contextvars(job_id=job.job_id, target_lang=job.config.target_language)

        log.info("pipeline_started", input_file=str(input_path))
        ACTIVE_JOBS.inc()

        try:
            # Stage 1: Extract audio from video
            await self._update_status(job, JobStatus.TRANSCRIBING, 5.0)
            with PIPELINE_STAGE_DURATION.labels("extract").time():
                raw_audio_path = await self.extractor.extract(input_path)
            log.info("audio_extracted", path=str(raw_audio_path))

            # Stage 2: Ambient separation (Phase 2)
            ambient_bytes = b""
            if settings.ambient_separation_enabled and raw_audio_path.exists():
                with PIPELINE_STAGE_DURATION.labels("ambient_separate").time():
                    raw_audio_bytes = raw_audio_path.read_bytes()
                    separation = await self.ambient_preserver.separate(raw_audio_bytes)
                    ambient_bytes = separation.ambient_bytes
                    if separation.has_significant_ambient:
                        log.info(
                            "ambient_detected",
                            level_db=separation.ambient_level_db,
                        )

            # Stage 3: Transcribe
            await self._update_status(job, JobStatus.TRANSCRIBING, 15.0)
            with PIPELINE_STAGE_DURATION.labels("transcribe").time():
                transcription = await self._transcribe(raw_audio_path, job.config)
            log.info(
                "transcription_completed",
                segments=len(transcription.segments),
                duration_ms=transcription.total_duration_ms,
            )

            # Stage 3.5: Boundary optimization (Phase 2)
            if settings.boundary_optimization_enabled:
                with PIPELINE_STAGE_DURATION.labels("boundary_optimize").time():
                    transcription = await self.boundary_optimizer.optimize(transcription)

            # Stage 4: Translate (batch + back-translation)
            await self._update_status(job, JobStatus.TRANSLATING, 35.0)
            with PIPELINE_STAGE_DURATION.labels("translate").time():
                translation = await self._translate(transcription, job.config)
            log.info(
                "translation_completed",
                segments=len(translation.segments),
                cache_hits=translation.cache_hits,
            )

            # Stage 5: Text-to-Speech (with emotion + pronunciation)
            await self._update_status(job, JobStatus.SYNTHESIZING, 55.0)
            with PIPELINE_STAGE_DURATION.labels("synthesize").time():
                tts_result = await self._synthesize(translation, job.config)
            log.info(
                "synthesis_completed",
                segments=len(tts_result.segments),
                total_bytes=tts_result.total_audio_bytes,
                provider=tts_result.provider_used,
            )

            # Stage 5.5: QC validation (Phase 2)
            if settings.qc_enabled:
                with PIPELINE_STAGE_DURATION.labels("qc_validate").time():
                    qc_result = await self.qc_controller.validate_pipeline_output(
                        tts_result.segments
                    )
                    # Track QC failures in metrics
                    for seg_result in qc_result.segments:
                        if seg_result.status == QCStatus.FAIL:
                            for issue in seg_result.issues:
                                reason = "silence" if "silence" in issue.lower() else \
                                         "length" if "length" in issue.lower() or "long" in issue.lower() or "short" in issue.lower() else \
                                         "size" if "small" in issue.lower() else "other"
                                QC_SEGMENT_FAILURES.labels(reason=reason).inc()

                    log.info(
                        "qc_complete",
                        overall=qc_result.overall_status,
                        pass_count=qc_result.pass_count,
                        warn_count=qc_result.warn_count,
                        fail_count=qc_result.fail_count,
                    )

            # Stage 6: Stitch audio
            await self._update_status(job, JobStatus.STITCHING, 85.0)
            with PIPELINE_STAGE_DURATION.labels("stitch").time():
                output_path = await self.stitcher.stitch(
                    tts_result.segments,
                    transcription.total_duration_ms,
                    job.job_id,
                )
            log.info("stitching_completed", output_path=str(output_path))

            # Stage 6.5: Ambient remix (Phase 2)
            if settings.ambient_separation_enabled and ambient_bytes:
                with PIPELINE_STAGE_DURATION.labels("ambient_remix").time():
                    dubbed_bytes = output_path.read_bytes()
                    remixed = await self.ambient_preserver.remix(dubbed_bytes, ambient_bytes)
                    output_path.write_bytes(remixed)
                    log.info("ambient_remixed")

            await self._update_status(job, JobStatus.COMPLETED, 100.0)
            JOBS_TOTAL.labels("completed").inc()
            log.info("pipeline_completed", output=str(output_path))
            return output_path

        except Exception as e:
            log.error(
                "pipeline_failed",
                stage=job.status,
                error_type=type(e).__name__,
                error=str(e),
            )
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            JOBS_TOTAL.labels("failed").inc()
            raise PipelineError(f"Pipeline failed at {job.status}: {e}") from e
        finally:
            ACTIVE_JOBS.dec()
            clear_contextvars()

    async def _transcribe(self, audio_path: Path, config: DubbingJobConfig) -> TranscriptionResult:
        provider = self.transcription_provider
        return await provider.transcribe(audio_path, config.source_language)

    async def _translate(
        self, transcription: TranscriptionResult, config: DubbingJobConfig
    ) -> TranslationResult:
        """
        Phase 2: Batch translation with cache + back-translation scoring.
        """
        provider = self.translation_providers[config.translation_provider]
        cache_hits = 0
        texts_to_translate = []
        cached_results = {}

        # Phase 1: Check cache for all segments
        for segment in transcription.segments:
            cache_key = f"{config.source_language}:{config.target_language}:{segment.original_text}"
            cached = await self.cache.get(cache_key)

            if cached:
                cached_results[segment.index] = cached
                cache_hits += 1
                TRANSLATION_CACHE_HITS.inc()
                log.debug("translation_cache_hit", segment_index=segment.index)
            else:
                texts_to_translate.append((segment.index, segment.original_text, cache_key))
                TRANSLATION_CACHE_MISSES.inc()

        # Phase 2: Batch translate ALL cache misses in ONE API call
        if texts_to_translate:
            indices, texts, keys = zip(*texts_to_translate)
            translated = await provider.translate_batch(
                list(texts), config.source_language, config.target_language
            )
            for idx, key, result in zip(indices, keys, translated):
                cached_results[idx] = result
                await self.cache.set(key, result)

        log.info(
            "batch_translation_complete",
            total=len(transcription.segments),
            cache_hits=cache_hits,
            api_calls=1 if texts_to_translate else 0,
        )

        # Apply translations to segments
        translated_segments = []
        for segment in transcription.segments:
            segment.translated_text = cached_results[segment.index]
            translated_segments.append(segment)

        # Phase 2: Back-translation quality scoring
        if settings.back_translation_enabled:
            for segment in translated_segments:
                bt_score = await self.back_translation_scorer.score(
                    original_text=segment.original_text,
                    translated_text=segment.translated_text,
                    source_lang=str(config.source_language),
                    target_lang=str(config.target_language),
                    translation_provider=provider,
                )
                BACK_TRANSLATION_SCORES.observe(bt_score.bleu_score)

                if bt_score.should_retry:
                    log.warning(
                        "back_translation_retry",
                        segment=segment.index,
                        bleu=bt_score.bleu_score,
                    )
                    BACK_TRANSLATION_RETRIES.inc()
                    # Try alternate provider
                    alt_provider_key = (
                        TranslationProvider.GOOGLE
                        if config.translation_provider == TranslationProvider.SARVAM
                        else TranslationProvider.SARVAM
                    )
                    alt_provider = self.translation_providers[alt_provider_key]
                    try:
                        segment.translated_text = await alt_provider.translate(
                            segment.original_text,
                            config.source_language,
                            config.target_language,
                        )
                    except Exception as e:
                        log.warning(
                            "back_translation_retry_failed",
                            segment=segment.index,
                            error=str(e),
                        )

        return TranslationResult(
            segments=translated_segments,
            source_language=config.source_language,
            target_language=config.target_language,
            provider_used=config.translation_provider,
            cache_hits=cache_hits,
        )

    async def _synthesize(
        self, translation: TranslationResult, config: DubbingJobConfig
    ) -> TTSResult:
        primary_provider = self.tts_providers[config.tts_provider]
        fallback_provider = self.tts_providers[TTSProvider.GTTS]

        synthesized_segments = []
        total_bytes = 0

        # Synthesize all segments concurrently for speed
        tasks = [
            self._synthesize_segment(
                segment, config, primary_provider, fallback_provider
            )
            for segment in translation.segments
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for segment, audio_bytes in zip(translation.segments, results):
            segment.audio_bytes = audio_bytes
            total_bytes += len(audio_bytes)
            synthesized_segments.append(segment)

        return TTSResult(
            segments=synthesized_segments,
            provider_used=config.tts_provider,
            total_audio_bytes=total_bytes,
        )

    async def _synthesize_segment(self, segment, config, primary, fallback) -> bytes:
        from vaaniflow.providers.tts.base import TTSSynthesisRequest

        # Phase 2: Pronunciation correction BEFORE TTS
        text_for_tts = segment.translated_text or segment.original_text
        if settings.pronunciation_correction_enabled:
            corrected_text, corrections = self.pronunciation_corrector.correct(text_for_tts)
            if corrections:
                log.debug(
                    "pronunciation_corrected",
                    segment=segment.index,
                    count=len(corrections),
                )
                text_for_tts = corrected_text

        # Phase 2: Emotion detection from original audio
        emotion = await self.emotion_preserver.detect(
            segment.audio_bytes or b""  # original audio bytes from extraction
        )
        EMOTION_DETECTIONS.labels(emotion=emotion.label.value).inc()

        request = TTSSynthesisRequest(
            text=text_for_tts,
            language=config.target_language,
            voice_id=config.voice_id,
            speaking_rate=emotion.speaking_rate,   # emotion-aware
            pitch=emotion.pitch_shift,             # emotion-aware
        )

        log.debug(
            "tts_with_emotion",
            segment=segment.index,
            emotion=emotion.label,
            speaking_rate=emotion.speaking_rate,
        )

        try:
            result = await primary.synthesize_with_logging(request)
            TTS_AUDIO_BYTES.labels(provider=primary.provider_name).observe(len(result.audio_bytes))
            return result.audio_bytes
        except Exception as e:
            log.warning(
                "tts_primary_failed_using_fallback",
                segment=segment.index,
                error=str(e),
            )
            PROVIDER_ERRORS.labels(
                provider=primary.provider_name,
                error_type=type(e).__name__,
            ).inc()
            result = await fallback.synthesize_with_logging(request)
            TTS_AUDIO_BYTES.labels(provider=fallback.provider_name).observe(len(result.audio_bytes))
            return result.audio_bytes

    @staticmethod
    async def _update_status(job: DubbingJob, status: JobStatus, progress: float):
        job.status = status
        job.progress_pct = progress
        from datetime import datetime
        job.updated_at = datetime.utcnow()
