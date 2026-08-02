"""
Unit tests for AudioExtractor.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from vaaniflow.audio.extractor import AudioExtractor, _resolve_ffmpeg
from vaaniflow.exceptions import AudioProcessingError


class TestAudioExtractor:

    def test_ffmpeg_resolution(self):
        """Test ffmpeg path resolution."""
        res = _resolve_ffmpeg()
        assert res is None or isinstance(res, str)

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(self):
        extractor = AudioExtractor()
        with pytest.raises(AudioProcessingError) as exc_info:
            await extractor.extract(Path("/nonexistent/file.wav"))
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_with_mocked_ffmpeg(self, tmp_path):
        input_file = tmp_path / "test.mp3"
        input_file.write_bytes(b"dummy audio content")

        extractor = AudioExtractor()
        with patch("vaaniflow.audio.extractor._resolve_ffmpeg", return_value="ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_res = MagicMock()
                mock_res.returncode = 0
                mock_run.return_value = mock_res

                fake_out = tmp_path / "extracted.wav"
                fake_out.write_bytes(b"fake wav")

                with patch("tempfile.mktemp", return_value=str(fake_out)):
                    res = await extractor.extract(input_file)
                    assert res.exists()
