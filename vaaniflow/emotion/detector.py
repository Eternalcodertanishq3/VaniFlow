"""
EmotionPreserver — detect emotional tone from audio and preserve it in TTS.

This is a unique original feature of VaaniFlow.
Most dubbing pipelines translate words. We preserve feelings.

Emotion detection pipeline:
  1. Extract audio features (pitch, energy, tempo) using librosa
  2. Classify into 5 emotions: neutral, happy, sad, angry, excited
  3. Map emotion -> TTS voice parameters (speaking_rate, pitch, stability)
  4. Inject parameters into TTS synthesis request

This is designed to be fast and local — no API call needed.
Runs in executor to avoid blocking the event loop.
"""
import asyncio
import structlog
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import io

log = structlog.get_logger(__name__)


class EmotionLabel(str, Enum):
    NEUTRAL  = "neutral"
    HAPPY    = "happy"
    SAD      = "sad"
    ANGRY    = "angry"
    EXCITED  = "excited"
    FEARFUL  = "fearful"


@dataclass
class EmotionResult:
    label: EmotionLabel
    confidence: float           # 0.0 - 1.0
    pitch_mean_hz: float        # mean pitch of segment
    energy_rms: float           # RMS energy
    tempo_bpm: float            # estimated speaking tempo
    # Derived TTS parameters
    speaking_rate: float        # inject into TTSSynthesisRequest
    pitch_shift: float          # inject into TTSSynthesisRequest
    tts_stability: float        # inject into ElevenLabs voice settings


# Emotion -> TTS parameter mapping
EMOTION_TTS_PARAMS = {
    EmotionLabel.NEUTRAL:  {"speaking_rate": 1.0,  "pitch_shift": 0.0,  "stability": 0.75},
    EmotionLabel.HAPPY:    {"speaking_rate": 1.1,  "pitch_shift": 0.15, "stability": 0.6},
    EmotionLabel.SAD:      {"speaking_rate": 0.88, "pitch_shift": -0.1, "stability": 0.85},
    EmotionLabel.ANGRY:    {"speaking_rate": 1.15, "pitch_shift": 0.05, "stability": 0.45},
    EmotionLabel.EXCITED:  {"speaking_rate": 1.2,  "pitch_shift": 0.2,  "stability": 0.4},
    EmotionLabel.FEARFUL:  {"speaking_rate": 1.05, "pitch_shift": 0.1,  "stability": 0.55},
}


class EmotionPreserver:
    """
    Detects emotion from original audio segments and maps it to TTS parameters.

    Usage:
        preserver = EmotionPreserver()
        emotion = await preserver.detect(segment_audio_bytes)
        tts_request.speaking_rate = emotion.speaking_rate
        tts_request.pitch = emotion.pitch_shift
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._librosa_available = None

    def _check_librosa(self) -> bool:
        if self._librosa_available is None:
            try:
                import librosa
                self._librosa_available = True
            except ImportError:
                log.warning(
                    "librosa_not_installed",
                    message="pip install librosa for emotion detection",
                    fallback="using neutral emotion for all segments",
                )
                self._librosa_available = False
        return self._librosa_available

    async def detect(self, audio_bytes: bytes) -> EmotionResult:
        """
        Detect emotion from raw audio bytes.
        Returns neutral if librosa unavailable or audio is too short.
        """
        if not self.enabled or not self._check_librosa():
            return self._neutral_result()

        if len(audio_bytes) < 500:
            return self._neutral_result()

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._detect_sync, audio_bytes)
        except Exception as e:
            log.warning("emotion_detection_failed", error=str(e))
            return self._neutral_result()

    def _detect_sync(self, audio_bytes: bytes) -> EmotionResult:
        """Synchronous emotion detection using librosa audio features."""
        import librosa
        import numpy as np

        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        if len(y) < sr * 0.3:  # Less than 0.3 seconds
            return self._neutral_result()

        # Feature extraction
        pitch_mean = self._extract_pitch(y, sr)
        energy_rms = float(np.sqrt(np.mean(y ** 2)))
        tempo_val = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_val[0]) if hasattr(tempo_val[0], '__float__') else float(tempo_val[0].item()) if hasattr(tempo_val[0], 'item') else 120.0

        # Spectral features for additional discrimination
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        # Rule-based emotion classification
        label = self._classify_emotion(
            pitch_mean=pitch_mean,
            energy_rms=energy_rms,
            tempo=tempo,
            spectral_centroid=spectral_centroid,
            zcr=zcr,
        )

        params = EMOTION_TTS_PARAMS[label]

        log.debug(
            "emotion_detected",
            label=label,
            pitch_hz=round(pitch_mean, 1),
            energy=round(energy_rms, 4),
            tempo_bpm=round(tempo, 1),
        )

        return EmotionResult(
            label=label,
            confidence=0.75,
            pitch_mean_hz=pitch_mean,
            energy_rms=energy_rms,
            tempo_bpm=tempo,
            speaking_rate=params["speaking_rate"],
            pitch_shift=params["pitch_shift"],
            tts_stability=params["stability"],
        )

    def _extract_pitch(self, y, sr) -> float:
        """Extract mean fundamental frequency (F0)."""
        import librosa
        import numpy as np
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"), sr=sr
            )
            voiced_f0 = f0[voiced_flag > 0.5] if f0 is not None else np.array([])
            return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        except Exception:
            return 0.0

    def _classify_emotion(
        self, pitch_mean: float, energy_rms: float,
        tempo: float, spectral_centroid: float, zcr: float
    ) -> EmotionLabel:
        """
        Rule-based emotion classifier.
        Based on prosodic feature research for speech emotion recognition.
        """
        HIGH_PITCH     = pitch_mean > 220
        LOW_PITCH      = pitch_mean < 130 and pitch_mean > 0
        HIGH_ENERGY    = energy_rms > 0.08
        LOW_ENERGY     = energy_rms < 0.02
        FAST_TEMPO     = tempo > 150
        SLOW_TEMPO     = tempo < 90
        HIGH_CENTROID  = spectral_centroid > 3000

        if HIGH_ENERGY and FAST_TEMPO and HIGH_PITCH:
            return EmotionLabel.EXCITED
        elif HIGH_ENERGY and HIGH_CENTROID and FAST_TEMPO:
            return EmotionLabel.ANGRY
        elif HIGH_PITCH and FAST_TEMPO and not HIGH_ENERGY:
            return EmotionLabel.HAPPY
        elif LOW_PITCH and SLOW_TEMPO and LOW_ENERGY:
            return EmotionLabel.SAD
        elif HIGH_PITCH and HIGH_ENERGY and not FAST_TEMPO:
            return EmotionLabel.FEARFUL
        else:
            return EmotionLabel.NEUTRAL

    def _neutral_result(self) -> EmotionResult:
        params = EMOTION_TTS_PARAMS[EmotionLabel.NEUTRAL]
        return EmotionResult(
            label=EmotionLabel.NEUTRAL, confidence=1.0,
            pitch_mean_hz=0.0, energy_rms=0.0, tempo_bpm=0.0,
            speaking_rate=params["speaking_rate"],
            pitch_shift=params["pitch_shift"],
            tts_stability=params["stability"],
        )
