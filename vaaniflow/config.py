"""
Application configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Centralized application configuration.
    All values can be overridden via environment variables.
    """

    # Application
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # TTS Provider Keys
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")

    # Translation Provider Keys
    google_translate_api_key: str = Field(default="", alias="GOOGLE_TRANSLATE_API_KEY")

    # AssemblyAI
    assemblyai_api_key: str = Field(default="", alias="ASSEMBLYAI_API_KEY")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")  # 24 hours

    # Provider settings
    provider_timeout_seconds: int = Field(default=30, alias="PROVIDER_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    # Output
    output_dir: str = Field(default="outputs", alias="OUTPUT_DIR")

    # Whisper model
    whisper_model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    # Phase 2: Feature toggles
    emotion_detection_enabled: bool = Field(default=True, alias="EMOTION_DETECTION_ENABLED")
    back_translation_enabled: bool = Field(default=True, alias="BACK_TRANSLATION_ENABLED")
    back_translation_threshold: float = Field(default=0.30, alias="BACK_TRANSLATION_THRESHOLD")
    boundary_optimization_enabled: bool = Field(default=True, alias="BOUNDARY_OPTIMIZATION_ENABLED")
    pronunciation_correction_enabled: bool = Field(default=True, alias="PRONUNCIATION_CORRECTION_ENABLED")
    ambient_separation_enabled: bool = Field(default=True, alias="AMBIENT_SEPARATION_ENABLED")
    qc_enabled: bool = Field(default=True, alias="QC_ENABLED")
    qc_max_silence_ratio: float = Field(default=0.7, alias="QC_MAX_SILENCE_RATIO")
    qc_max_length_ratio: float = Field(default=3.0, alias="QC_MAX_LENGTH_RATIO")

    # Phase 3: Showcase features
    lipsync_export_enabled: bool = Field(default=False, alias="LIPSYNC_EXPORT_ENABLED")
    code_switch_normalization_enabled: bool = Field(default=True, alias="CODE_SWITCH_NORMALIZATION_ENABLED")
    subtitle_generation_enabled: bool = Field(default=True, alias="SUBTITLE_GENERATION_ENABLED")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Singleton settings instance
settings = Settings()
