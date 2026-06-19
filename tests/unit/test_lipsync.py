"""
Unit tests for LipSyncExporter — the multi-modal pipeline placeholder.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from vaaniflow.lipsync import LipSyncExporter, LipSyncManifest, LipSyncSegment
from vaaniflow.models import AudioSegment


@pytest.fixture
def segments():
    """Sample dubbed audio segments with timing."""
    return [
        AudioSegment(
            index=0, start_ms=0, end_ms=2500,
            duration_ms=2500, original_text="Hello world",
            translated_text="नमस्ते दुनिया",
        ),
        AudioSegment(
            index=1, start_ms=2500, end_ms=5000,
            duration_ms=2500, original_text="How are you",
            translated_text="आप कैसे हैं",
        ),
    ]


@pytest.fixture
def exporter(tmp_path):
    return LipSyncExporter(enabled=True, output_dir=str(tmp_path))


@pytest.fixture
def disabled_exporter(tmp_path):
    return LipSyncExporter(enabled=False, output_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_disabled_returns_none(disabled_exporter, segments):
    """Disabled exporter should return None immediately."""
    result = await disabled_exporter.export(
        segments=segments, job_id="test-123",
        dubbed_audio_path=Path("/tmp/dubbed.wav"),
        source_language="en", target_language="hi",
        total_duration_ms=5000.0,
    )
    assert result is None


@pytest.mark.asyncio
async def test_export_creates_manifest(exporter, segments, tmp_path):
    """Exporter should create a JSON manifest file."""
    result = await exporter.export(
        segments=segments, job_id="test-456",
        dubbed_audio_path=Path("/tmp/dubbed.wav"),
        source_language="en", target_language="hi",
        total_duration_ms=5000.0,
    )
    assert result is not None
    assert result.exists()
    assert result.suffix == ".json"
    assert "test-456" in result.name


@pytest.mark.asyncio
async def test_manifest_content_structure(exporter, segments, tmp_path):
    """Manifest JSON should have correct structure and data."""
    result = await exporter.export(
        segments=segments, job_id="job-789",
        dubbed_audio_path=Path("/tmp/dubbed.wav"),
        source_language="en", target_language="hi",
        total_duration_ms=5000.0,
    )
    with open(result, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["job_id"] == "job-789"
    assert data["source_language"] == "en"
    assert data["target_language"] == "hi"
    assert data["total_duration_ms"] == 5000.0
    assert len(data["segments"]) == 2
    assert data["segments"][0]["original_text"] == "Hello world"
    assert data["segments"][0]["translated_text"] == "नमस्ते दुनिया"
    assert data["segments"][0]["start_ms"] == 0
    assert data["segments"][0]["end_ms"] == 2500


@pytest.mark.asyncio
async def test_manifest_includes_emotion(exporter, segments, tmp_path):
    """Manifest should include emotion metadata when provided."""
    mock_emotion = MagicMock()
    mock_emotion.label.value = "happy"
    mock_emotion.speaking_rate = 1.1
    emotions = {0: mock_emotion}

    result = await exporter.export(
        segments=segments, job_id="emo-test",
        dubbed_audio_path=Path("/tmp/dubbed.wav"),
        source_language="en", target_language="hi",
        total_duration_ms=5000.0,
        emotions=emotions,
    )
    with open(result, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["segments"][0]["emotion_label"] == "happy"
    assert data["segments"][0]["speaking_rate"] == 1.1
    assert data["segments"][1]["emotion_label"] is None


@pytest.mark.asyncio
async def test_manifest_default_renderer(exporter, segments, tmp_path):
    """Default renderer should be wav2lip."""
    result = await exporter.export(
        segments=segments, job_id="renderer-test",
        dubbed_audio_path=Path("/tmp/dubbed.wav"),
        source_language="en", target_language="hi",
        total_duration_ms=5000.0,
    )
    with open(result, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["renderer"] == "wav2lip"


def test_lipsync_manifest_dataclass():
    """LipSyncManifest should serialize cleanly."""
    manifest = LipSyncManifest(
        job_id="dc-test",
        source_language="en",
        target_language="ta",
        total_duration_ms=3000.0,
        dubbed_audio_path="/out/test.wav",
    )
    d = manifest.to_dict()
    assert d["job_id"] == "dc-test"
    assert d["segments"] == []
    assert d["renderer"] == "wav2lip"
