"""
QC result models — every segment gets a quality score before stitching.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class QCStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class SegmentQCResult(BaseModel):
    segment_index: int
    status: QCStatus
    silence_ratio: float = Field(description="Ratio of silence to total audio (0-1)")
    length_ratio: float = Field(description="TTS length / original length ratio")
    issues: list[str] = Field(default_factory=list)
    should_retry: bool = False
    fallback_provider: Optional[str] = None


class PipelineQCResult(BaseModel):
    overall_status: QCStatus
    segments: list[SegmentQCResult]
    pass_count: int
    warn_count: int
    fail_count: int
    retry_segments: list[int]  # segment indices that need retry


class QCConfig(BaseModel):
    max_silence_ratio: float = Field(default=0.7, description="Fail if >70% silence")
    max_length_ratio: float = Field(default=3.0, description="Warn if TTS is 3x original")
    min_length_ratio: float = Field(default=0.3, description="Warn if TTS is <30% original")
    min_audio_bytes: int = Field(default=100, description="Fail if audio < 100 bytes")
    auto_retry_on_fail: bool = True
