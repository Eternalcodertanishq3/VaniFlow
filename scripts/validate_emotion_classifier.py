"""
Validation script for EmotionPreserver's rule-based classifier.
Run manually: python scripts/validate_emotion_classifier.py

Requires a small labeled sample set (RAVDESS subset recommended:
https://zenodo.org/record/1188976 — download ~15-20 clips covering
each emotion). Populate VALIDATION_SET below with real paths, then run.

This is a standalone diagnostic, not part of the test suite.
"""
import asyncio
from pathlib import Path
from collections import defaultdict

from vaaniflow.emotion.detector import EmotionPreserver, EmotionLabel

VALIDATION_SET = [
    # ("samples/ravdess_03-01-05-01-01-01-01.wav", EmotionLabel.ANGRY),
    # ("samples/ravdess_03-01-03-01-01-01-01.wav", EmotionLabel.HAPPY),
    # Add real labeled file paths here before running.
]


async def main():
    if not VALIDATION_SET:
        print("No validation samples configured. Download a small labeled subset "
              "from RAVDESS and populate VALIDATION_SET before running.\n")
        return

    preserver = EmotionPreserver(enabled=True)
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0

    for file_path, ground_truth in VALIDATION_SET:
        audio_bytes = Path(file_path).read_bytes()
        result = await preserver.detect(audio_bytes)
        confusion[ground_truth][result.label] += 1
        if result.label == ground_truth:
            correct += 1
        print(f"{file_path}: predicted={result.label.value}, actual={ground_truth.value}")

    accuracy = correct / len(VALIDATION_SET) * 100
    print(f"\nOverall accuracy: {accuracy:.1f}% ({correct}/{len(VALIDATION_SET)})")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    for actual, predictions in confusion.items():
        print(f"  {actual.value}: {dict(predictions)}")


if __name__ == "__main__":
    asyncio.run(main())
