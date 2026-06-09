"""
VaaniFlow Pipeline Orchestrator.
This is the heart of the system — orchestrates all providers
in the correct sequence with proper error handling at each stage.
"""
import asyncio
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

log = structlog.get_logger(__name__)


class VaaniFlowPipeline:
    """
    Main dubbing pipeline orchestrator.
    Manages the full flow: extract → transcribe → translate → synthesize → stitch
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

    async def run(self, job: DubbingJob, input_path: Path) -> Path:
        """
        Execute the full dubbing pipeline for a job.
        Each stage updates job.status and logs progress.
        """
        # Bind job context to all logs in this coroutine
        bind_contextvars(job_id=job.job_id, target_lang=job.config.target_language)

        log.info("pipeline_started", input_file=str(input_path))

        try:
            # Stage 1: Extract audio from video
            await self._update_status(job, JobStatus.TRANSCRIBING, 5.0)
            raw_audio_path = await self.extractor.extract(input_path)
            log.info("audio_extracted", path=str(raw_audio_path))

            # Stage 2: Transcribe
            await self._update_status(job, JobStatus.TRANSCRIBING, 15.0)
            transcription = await self._transcribe(raw_audio_path, job.config)
            log.info(
                "transcription_completed",
                segments=len(transcription.segments),
                duration_ms=transcription.total_duration_ms,
            )

            # Stage 3: Translate
            await self._update_status(job, JobStatus.TRANSLATING, 35.0)
            translation = await self._translate(transcription, job.config)
            log.info(
                "translation_completed",
                segments=len(translation.segments),
                cache_hits=translation.cache_hits,
            )

            # Stage 4: Text-to-Speech
            await self._update_status(job, JobStatus.SYNTHESIZING, 55.0)
            tts_result = await self._synthesize(translation, job.config)
            log.info(
                "synthesis_completed",
                segments=len(tts_result.segments),
                total_bytes=tts_result.total_audio_bytes,
                provider=tts_result.provider_used,
            )

            # Stage 5: Stitch audio
            await self._update_status(job, JobStatus.STITCHING, 85.0)
            output_path = await self.stitcher.stitch(
                tts_result.segments,
                transcription.total_duration_ms,
                job.job_id,
            )
            log.info("stitching_completed", output_path=str(output_path))

            await self._update_status(job, JobStatus.COMPLETED, 100.0)
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
            raise PipelineError(f"Pipeline failed at {job.status}: {e}") from e
        finally:
            clear_contextvars()

    async def _transcribe(self, audio_path: Path, config: DubbingJobConfig) -> TranscriptionResult:
        provider = self.transcription_provider
        return await provider.transcribe(audio_path, config.source_language)

    async def _translate(
        self, transcription: TranscriptionResult, config: DubbingJobConfig
    ) -> TranslationResult:
        provider = self.translation_providers[config.translation_provider]
        cache_hits = 0
        translated_segments = []

        # Translate each segment — check cache first
        for segment in transcription.segments:
            cache_key = f"{config.source_language}:{config.target_language}:{segment.original_text}"
            cached = await self.cache.get(cache_key)

            if cached:
                segment.translated_text = cached
                cache_hits += 1
                log.debug("translation_cache_hit", segment_index=segment.index)
            else:
                segment.translated_text = await provider.translate(
                    segment.original_text,
                    config.source_language,
                    config.target_language,
                )
                await self.cache.set(cache_key, segment.translated_text)

            translated_segments.append(segment)

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
        request = TTSSynthesisRequest(
            text=segment.translated_text,
            language=config.target_language,
            voice_id=config.voice_id,
        )
        try:
            result = await primary.synthesize_with_logging(request)
            return result.audio_bytes
        except Exception as e:
            log.warning(
                "tts_primary_failed_using_fallback",
                segment=segment.index,
                error=str(e),
            )
            result = await fallback.synthesize_with_logging(request)
            return result.audio_bytes

    @staticmethod
    async def _update_status(job: DubbingJob, status: JobStatus, progress: float):
        job.status = status
        job.progress_pct = progress
        from datetime import datetime
        job.updated_at = datetime.utcnow()
