"""
Neural emotion detection using wav2vec2 + emotion classification.

Upgrades the rule-based EmotionPreserver with a pre-trained neural classifier.
Falls back to rule-based detection if the model is unavailable.

Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
  - Trained on RAVDESS and other emotional speech datasets
  - 7 classes: angry, calm, disgust, fearful, happy, neutral, sad
  - Runs on CPU at ~100ms per 3-second clip
  - ~1.2GB model (downloaded once, cached by HuggingFace)

Architecture:
  Primary: wav2vec2 neural classifier (high accuracy, needs transformers + torch)
  Fallback: Rule-based librosa feature analysis (always available)
"""

import asyncio
import importlib.util
import io

import structlog

from vaaniflow.emotion.detector import (
    EMOTION_TTS_PARAMS,
    EmotionLabel,
    EmotionPreserver,
    EmotionResult,
)

log = structlog.get_logger(__name__)

# Map wav2vec2 model output labels to our EmotionLabel enum
_NEURAL_LABEL_MAP: dict[str, EmotionLabel] = {
    "angry": EmotionLabel.ANGRY,
    "ang": EmotionLabel.ANGRY,
    "calm": EmotionLabel.NEUTRAL,
    "disgust": EmotionLabel.ANGRY,  # closest mapping
    "fearful": EmotionLabel.FEARFUL,
    "fear": EmotionLabel.FEARFUL,
    "happy": EmotionLabel.HAPPY,
    "hap": EmotionLabel.HAPPY,
    "neutral": EmotionLabel.NEUTRAL,
    "neu": EmotionLabel.NEUTRAL,
    "sad": EmotionLabel.SAD,
    "sadness": EmotionLabel.SAD,
    "surprise": EmotionLabel.EXCITED,
    "excited": EmotionLabel.EXCITED,
    "exc": EmotionLabel.EXCITED,
}


class NeuralEmotionPreserver:
    """
    Neural emotion detector with rule-based fallback.

    Uses HuggingFace transformers audio-classification pipeline
    with a wav2vec2 model for emotion recognition. Falls back to
    the existing rule-based EmotionPreserver if transformers/torch
    are not installed.

    Usage (drop-in replacement for EmotionPreserver):
        preserver = NeuralEmotionPreserver(enabled=True)
        emotion = await preserver.detect(segment_audio_bytes)
        tts_request.speaking_rate = emotion.speaking_rate
    """

    MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    def __init__(self, enabled: bool = True, fallback_to_rule_based: bool = True):
        self.enabled = enabled
        self.fallback_to_rule_based = fallback_to_rule_based
        self._classifier = None
        self._model_load_attempted = False
        self._model_load_failed = False
        self._rule_based = EmotionPreserver(enabled=enabled)

    def _check_dependencies(self) -> bool:
        """Check if transformers and torch are installed."""
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _load_model(self):
        """Lazy-load the wav2vec2 classifier. Called once on first use."""
        if self._model_load_attempted:
            return self._classifier

        self._model_load_attempted = True

        if not self._check_dependencies():
            log.info(
                "neural_emotion_dependencies_missing",
                message="Install transformers and torch for neural emotion detection",
                fallback="rule-based",
            )
            self._model_load_failed = True
            return None

        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "audio-classification",
                model=self.MODEL_ID,
                device=-1,  # CPU — safe default, GPU auto-detection can be added
            )
            log.info("neural_emotion_model_loaded", model=self.MODEL_ID)
        except Exception as e:
            log.warning(
                "neural_emotion_model_load_failed",
                error=str(e),
                fallback="rule-based",
            )
            self._model_load_failed = True

        return self._classifier

    async def detect(self, audio_bytes: bytes) -> EmotionResult:
        """
        Detect emotion from raw audio bytes.

        Primary: wav2vec2 neural classifier
        Fallback: Rule-based librosa feature analysis

        Returns EmotionResult with label, confidence, and TTS parameters.
        """
        if not self.enabled or len(audio_bytes) < 1024:
            return self._rule_based._neutral_result()

        # Try neural detection first
        classifier = self._load_model()
        if classifier is None:
            if self.fallback_to_rule_based:
                return await self._rule_based.detect(audio_bytes)
            return self._rule_based._neutral_result()

        try:
            return await asyncio.to_thread(
                self._detect_neural_sync, audio_bytes, classifier
            )
        except Exception as e:
            log.warning("neural_emotion_detection_failed", error=str(e))
            if self.fallback_to_rule_based:
                return await self._rule_based.detect(audio_bytes)
            return self._rule_based._neutral_result()

    def _detect_neural_sync(self, audio_bytes: bytes, classifier) -> EmotionResult:
        """Synchronous neural emotion detection. Runs in thread pool."""
        import librosa
        import numpy as np

        # Load audio at 16kHz mono (wav2vec2 requirement)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        if len(y) < sr * 0.3:  # Less than 0.3 seconds — too short
            return self._rule_based._neutral_result()

        # Run wav2vec2 inference
        results = classifier({"raw": y, "sampling_rate": sr})

        # Get top prediction
        top = results[0]
        raw_label = top["label"].lower().strip()
        confidence = float(top["score"])

        # Map neural label to our enum
        label = _NEURAL_LABEL_MAP.get(raw_label, EmotionLabel.NEUTRAL)

        # Get TTS parameters from the emotion-to-TTS mapping
        params = EMOTION_TTS_PARAMS[label]

        # Scale TTS parameters by confidence:
        # High confidence → stronger emotion expression
        # Low confidence → closer to neutral
        confidence_scale = 0.5 + (confidence * 0.5)  # Maps [0,1] → [0.5, 1.0]

        speaking_rate = 1.0 + (params["speaking_rate"] - 1.0) * confidence_scale
        pitch_shift = params["pitch_shift"] * confidence_scale
        stability = params["stability"]

        # Extract basic acoustic features for the result dataclass
        pitch_mean = self._quick_pitch(y, sr)
        energy_rms = float(np.sqrt(np.mean(y**2)))

        log.info(
            "neural_emotion_detected",
            label=label.value,
            confidence=round(confidence, 3),
            raw_label=raw_label,
            speaking_rate=round(speaking_rate, 3),
        )

        return EmotionResult(
            label=label,
            confidence=confidence,
            pitch_mean_hz=pitch_mean,
            energy_rms=energy_rms,
            tempo_bpm=0.0,  # Not computed by neural model
            speaking_rate=speaking_rate,
            pitch_shift=pitch_shift,
            tts_stability=stability,
        )

    @staticmethod
    def _quick_pitch(y, sr: int) -> float:
        """Quick F0 estimate using librosa pyin."""
        try:
            import librosa
            import numpy as np

            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            voiced_f0 = f0[voiced_flag > 0.5] if f0 is not None else np.array([])
            return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        except Exception:
            return 0.0
