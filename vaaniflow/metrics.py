"""
Prometheus metrics definitions for VaaniFlow.
Defined at library level to avoid circular imports (pipeline -> api).
The API route simply serves the Prometheus registry.
"""
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
)

# Job lifecycle
JOBS_TOTAL = Counter(
    "vaaniflow_jobs_total",
    "Total dubbing jobs created",
    ["status"],
)

ACTIVE_JOBS = Gauge(
    "vaaniflow_active_jobs",
    "Number of currently running pipeline jobs",
)

# Pipeline stage timing
PIPELINE_STAGE_DURATION = Histogram(
    "vaaniflow_pipeline_stage_duration_seconds",
    "Duration of each pipeline stage",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Translation cache
TRANSLATION_CACHE_HITS = Counter(
    "vaaniflow_translation_cache_hits_total",
    "Number of translation cache hits",
)

TRANSLATION_CACHE_MISSES = Counter(
    "vaaniflow_translation_cache_misses_total",
    "Number of translation cache misses",
)

# Provider errors
PROVIDER_ERRORS = Counter(
    "vaaniflow_provider_errors_total",
    "Provider errors by type and provider",
    ["provider", "error_type"],
)

# TTS output size
TTS_AUDIO_BYTES = Histogram(
    "vaaniflow_tts_audio_bytes",
    "Size of synthesized TTS audio in bytes",
    ["provider"],
    buckets=[1000, 10000, 50000, 100000, 500000, 1000000],
)

# Quality control
QC_SEGMENT_FAILURES = Counter(
    "vaaniflow_qc_segment_failures_total",
    "QC failures by reason",
    ["reason"],
)

# Emotion detection
EMOTION_DETECTIONS = Counter(
    "vaaniflow_emotion_detections_total",
    "Emotion detections by label",
    ["emotion"],
)

# Back-translation quality
BACK_TRANSLATION_SCORES = Histogram(
    "vaaniflow_back_translation_bleu_scores",
    "BLEU scores from back-translation quality checks",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

BACK_TRANSLATION_RETRIES = Counter(
    "vaaniflow_back_translation_retries_total",
    "Number of translation retries due to low BLEU scores",
)
