"""
File upload validation for dubbing job creation.
Enforces file size limits, format whitelist, and content-type checks.
"""
from pathlib import Path

from fastapi import HTTPException, UploadFile
import structlog

from vaaniflow.config import settings

log = structlog.get_logger(__name__)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = set(
    ext.strip().lower()
    for ext in settings.allowed_upload_formats.split(",")
    if ext.strip()
)

# Allowed content types for audio/video uploads
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/ogg", "audio/flac", "audio/x-flac", "audio/m4a", "audio/mp4",
    "audio/webm", "audio/aac",
    "video/mp4", "video/webm", "video/x-matroska", "video/ogg",
    "video/quicktime", "video/x-msvideo",
    "application/octet-stream",  # Accept generic binary (common for programmatic uploads)
}

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024


async def validate_upload(file: UploadFile) -> bytes:
    """
    Validate an uploaded file and return its content.
    
    Checks:
    1. Filename exists and has an allowed extension
    2. Content-type is a known audio/video type
    3. File size is within the configured limit
    
    Returns:
        File content as bytes
        
    Raises:
        HTTPException(400): Invalid format or missing filename
        HTTPException(413): File too large
    """
    # Check filename exists
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check content type
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        log.warning(
            "upload_content_type_rejected",
            filename=file.filename,
            content_type=content_type,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{content_type}'. Expected audio or video file.",
        )

    # Read and check file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f}MB). Maximum: {settings.max_upload_size_mb}MB",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    log.info(
        "upload_validated",
        filename=file.filename,
        size_bytes=len(content),
        content_type=content_type,
    )

    return content
