"""
All data models using Pydantic v2.
Sarvam wants typed Python — every input/output/config must be typed.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
import uuid


class SupportedLanguage(str, Enum):
    """10+ Indian languages supported by VaaniFlow."""
    HINDI = "hi"
    BENGALI = "bn"
    TELUGU = "te"
    MARATHI = "mr"
    TAMIL = "ta"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    ODIA = "or"
    ENGLISH = "en"


class TTSProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    SARVAM = "sarvam"
    GTTS = "gtts"  # fallback


class TranslationProvider(str, Enum):
    GOOGLE = "google"
    SARVAM = "sarvam"


class TranscriptionProvider(str, Enum):
    WHISPER = "whisper"
    ASSEMBLY = "assemblyai"


class JobStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    STITCHING = "stitching"
    COMPLETED = "completed"
    FAILED = "failed"


class DubbingJobConfig(BaseModel):
    """
    Configuration for a single dubbing job.
    Pydantic v2 with full validation.
    """
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH
    target_language: SupportedLanguage
    tts_provider: TTSProvider = TTSProvider.SARVAM
    translation_provider: TranslationProvider = TranslationProvider.SARVAM
    transcription_provider: TranscriptionProvider = TranscriptionProvider.WHISPER
    preserve_timing: bool = True
    voice_id: Optional[str] = None
    max_retries: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    @field_validator("target_language")
    @classmethod
    def source_and_target_must_differ(cls, v, info):
        if "source_language" in info.data and v == info.data["source_language"]:
            raise ValueError("Source and target language must be different")
        return v


class AudioSegment(BaseModel):
    """A single transcribed/translated audio segment with timing."""
    index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    original_text: str
    translated_text: Optional[str] = None
    audio_bytes: Optional[bytes] = None

    model_config = {"arbitrary_types_allowed": True}


class TranscriptionResult(BaseModel):
    """Output from transcription provider."""
    segments: list[AudioSegment]
    source_language: str
    total_duration_ms: float
    provider_used: TranscriptionProvider


class TranslationResult(BaseModel):
    """Output from translation provider."""
    segments: list[AudioSegment]
    source_language: SupportedLanguage
    target_language: SupportedLanguage
    provider_used: TranslationProvider
    cache_hits: int = 0


class TTSResult(BaseModel):
    """Output from TTS provider."""
    segments: list[AudioSegment]
    provider_used: TTSProvider
    total_audio_bytes: int


class DubbingJob(BaseModel):
    """Full dubbing job state — stored in Redis."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    config: DubbingJobConfig
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    progress_pct: float = 0.0


class DubbingJobRequest(BaseModel):
    """API request to create a dubbing job."""
    target_language: SupportedLanguage
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH
    tts_provider: TTSProvider = TTSProvider.SARVAM
    voice_id: Optional[str] = None

    def to_config(self) -> DubbingJobConfig:
        """Convert API request to internal job config."""
        return DubbingJobConfig(
            source_language=self.source_language,
            target_language=self.target_language,
            tts_provider=self.tts_provider,
            voice_id=self.voice_id,
        )


class DubbingJobResponse(BaseModel):
    """API response for job creation/status."""
    job_id: str
    status: JobStatus
    progress_pct: float
    output_url: Optional[str] = None
    error: Optional[str] = None
