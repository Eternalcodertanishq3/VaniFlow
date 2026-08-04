"""
Unit tests for file upload validation.
"""
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile


class TestUploadValidation:
    """Tests for upload file validation."""

    @pytest.fixture(autouse=True)
    def mock_settings(self):
        """Mock settings for all tests."""
        with patch("api.middleware.upload_validation.settings") as mock:
            mock.max_upload_size_mb = 100
            mock.allowed_upload_formats = ".mp3,.mp4,.wav,.webm,.ogg,.m4a,.flac,.mkv"
            yield mock

    def _make_upload(self, filename: str, content: bytes, content_type: str = "audio/wav") -> UploadFile:
        """Create a mock UploadFile."""
        file = UploadFile(
            filename=filename,
            file=BytesIO(content),
            headers={"content-type": content_type},
        )
        return file

    @pytest.mark.asyncio
    async def test_valid_wav_upload(self):
        """Valid WAV file should pass validation."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("test.wav", b"fake_audio" * 100, "audio/wav")
        content = await validate_upload(file)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_valid_mp4_upload(self):
        """Valid MP4 file should pass validation."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("video.mp4", b"fake_video" * 100, "video/mp4")
        content = await validate_upload(file)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_reject_unsupported_extension(self):
        """Files with unsupported extensions should be rejected."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("malware.exe", b"bad_content", "application/octet-stream")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(file)
        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_no_filename(self):
        """Files without a filename should be rejected."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("", b"content", "audio/wav")
        file.filename = ""
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(file)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_reject_oversized_file(self):
        """Files exceeding the size limit should be rejected."""
        from api.middleware.upload_validation import validate_upload
        # Create content larger than the limit
        huge_content = b"x" * (101 * 1024 * 1024)  # 101MB
        file = self._make_upload("large.wav", huge_content, "audio/wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(file)
        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reject_empty_file(self):
        """Empty files should be rejected."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("empty.wav", b"", "audio/wav")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(file)
        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_reject_bad_content_type(self):
        """Files with non-audio/video content types should be rejected."""
        from api.middleware.upload_validation import validate_upload
        file = self._make_upload("file.wav", b"content" * 10, "text/html")
        with pytest.raises(HTTPException) as exc_info:
            await validate_upload(file)
        assert exc_info.value.status_code == 400
        assert "content type" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_accept_all_allowed_formats(self):
        """All allowed formats should pass validation."""
        from api.middleware.upload_validation import validate_upload
        for ext in [".mp3", ".mp4", ".wav", ".webm", ".ogg", ".m4a", ".flac", ".mkv"]:
            file = self._make_upload(f"test{ext}", b"content" * 10, "application/octet-stream")
            content = await validate_upload(file)
            assert len(content) > 0, f"Format {ext} should be accepted"
