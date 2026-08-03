"""Emotion detection and preservation for VaaniFlow."""

from vaaniflow.emotion.detector import EmotionLabel, EmotionPreserver, EmotionResult
from vaaniflow.emotion.neural_detector import NeuralEmotionPreserver

# Alias for clarity — NeuralEmotionPreserver is now language-aware
LanguageAwareEmotionPreserver = NeuralEmotionPreserver

__all__ = [
    "EmotionLabel",
    "EmotionPreserver",
    "EmotionResult",
    "NeuralEmotionPreserver",
    "LanguageAwareEmotionPreserver",
]
