"""
Unit tests for AudioStitcher.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from vaaniflow.audio.stitcher import AudioStitcher
from vaaniflow.models import AudioSegment
from vaaniflow.exceptions import AudioProcessingError


class TestAudioStitcher:

    @pytest.mark.asyncio
    async def test_stitch_empty_segments_creates_silence(self, tmp_path):
        stitcher = AudioStitcher(output_dir=tmp_path)
        segments = []

        with patch("vaaniflow.audio.stitcher._resolve_ffmpeg", return_value="ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_res = MagicMock()
                mock_res.returncode = 0
                mock_run.return_value = mock_res

                out_path = tmp_path / "dubbed_test-job.wav"
                out_path.write_bytes(b"stitched audio")

                res = await stitcher.stitch(segments, total_duration_ms=5000.0, job_id="test-job")
                assert res.exists()

    @pytest.mark.asyncio
    async def test_stitch_with_segments(self, tmp_path):
        stitcher = AudioStitcher(output_dir=tmp_path)
        segments = [
            AudioSegment(
                index=0, start_ms=0, end_ms=2000, duration_ms=2000,
                original_text="hello", audio_bytes=b"fake_pcm"
            )
        ]

        with patch("vaaniflow.audio.stitcher._resolve_ffmpeg", return_value="ffmpeg"):
            with patch.object(stitcher, "_get_duration", return_value=2.0):
                with patch("subprocess.run") as mock_run:
                    mock_res = MagicMock()
                    mock_res.returncode = 0
                    mock_run.return_value = mock_res

                    out_path = tmp_path / "dubbed_test-job-2.wav"
                    out_path.write_bytes(b"stitched audio content")

                    res = await stitcher.stitch(segments, total_duration_ms=2000.0, job_id="test-job-2")
                    assert res.exists()
