import pytest
from vaaniflow.subtitles.generator import SubtitleGenerator, _format_srt_timestamp, _format_vtt_timestamp
from vaaniflow.models import AudioSegment

@pytest.fixture
def segments():
    return [
        AudioSegment(index=0, start_ms=0, end_ms=2500, duration_ms=2500,
                     original_text="Hello world", translated_text="नमस्ते दुनिया"),
        AudioSegment(index=1, start_ms=2600, end_ms=5000, duration_ms=2400,
                     original_text="How are you", translated_text="आप कैसे हैं"),
    ]

def test_srt_timestamp_format():
    assert _format_srt_timestamp(0) == "00:00:00,000"
    assert _format_srt_timestamp(1500) == "00:00:01,500"
    assert _format_srt_timestamp(65000) == "00:01:05,000"

def test_vtt_timestamp_format():
    assert _format_vtt_timestamp(1500) == "00:00:01.500"

def test_generate_srt(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_srt(segments, job_id="test-job", use_translated=True)
    assert path is not None and path.exists()
    content = path.read_text(encoding="utf-8")
    assert "नमस्ते दुनिया" in content
    assert "00:00:00,000 --> 00:00:02,500" in content

def test_generate_vtt(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_vtt(segments, job_id="test-job")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "आप कैसे हैं" in content

def test_disabled_returns_none(segments, tmp_path):
    gen = SubtitleGenerator(enabled=False, output_dir=str(tmp_path))
    assert gen.generate_srt(segments, "x") is None
    assert gen.generate_vtt(segments, "x") is None

def test_use_original_text(segments, tmp_path):
    gen = SubtitleGenerator(enabled=True, output_dir=str(tmp_path))
    path = gen.generate_srt(segments, job_id="orig-test", use_translated=False)
    content = path.read_text(encoding="utf-8")
    assert "Hello world" in content
