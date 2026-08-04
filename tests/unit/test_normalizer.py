"""
Unit tests for AudioNormalizer.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaaniflow.audio.normalizer import AudioNormalizer
from vaaniflow.exceptions import AudioProcessingError


class TestAudioNormalizer:

    @pytest.mark.asyncio
    async def test_normalize_nonexistent_file(self):
        with pytest.raises(AudioProcessingError):
            await AudioNormalizer.normalize_volume(Path("/nonexistent/file.wav"))

    @pytest.mark.asyncio
    async def test_normalize_volume_mocked_ffmpeg(self, tmp_path):
        in_file = tmp_path / "in.wav"
        in_file.write_bytes(b"dummy wav")
        out_file = tmp_path / "out.wav"

        with patch("vaaniflow.audio.normalizer._resolve_ffmpeg", return_value="ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_res = MagicMock()
                mock_res.returncode = 0
                mock_run.return_value = mock_res

                res = await AudioNormalizer.normalize_volume(in_file, out_file)
                assert res == out_file

    @pytest.mark.asyncio
    async def test_convert_sample_rate_mocked_ffmpeg(self, tmp_path):
        in_file = tmp_path / "in.wav"
        in_file.write_bytes(b"dummy wav")
        out_file = tmp_path / "resampled.wav"

        with patch("vaaniflow.audio.normalizer._resolve_ffmpeg", return_value="ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_res = MagicMock()
                mock_res.returncode = 0
                mock_run.return_value = mock_res

                res = await AudioNormalizer.convert_sample_rate(in_file, out_file, target_rate=16000)
                assert res == out_file
