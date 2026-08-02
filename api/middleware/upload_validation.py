"""
File upload validation for dubbing job creation.
Enforces file size limits, format whitelist, and content-type checks.
"""
from pathlib import Path

from fastapi import HTTPException, UploadFile
import structlog

from vaaniflow.config import settings

log = structlog.get_logger(__name__)

# Allowed content types for audio/video uploads
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/ogg", "audio/flac", "audio/x-flac", "audio/m4a", "audio/mp4",
    "audio/webm", "audio/aac",
    "video/mp4", "video/webm", "video/x-matroska", "video/ogg",
    "video/quicktime", "video/x-msvideo",
    "application/octet-stream",  # Accept generic binary (common for programmatic uploads)
}


def get_allowed_extensions() -> set[str]:
    """Dynamically get allowed extensions from settings."""
    return set(
        ext.strip().lower()
        for ext in settings.allowed_upload_formats.split(",")
        if ext.strip()
    )


def get_max_upload_bytes() -> int:
    """Dynamically get max upload bytes from settings."""
    return settings.max_upload_size_mb * 1024 * 1024


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
    allowed_extensions = get_allowed_extensions()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
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

    # Stream file in 1MB chunks to check size limit efficiently before accumulating memory
    chunks = []
    total_size = 0
    max_upload_bytes = get_max_upload_bytes()
    chunk_size = 1024 * 1024  # 1MB chunk

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({total_size / 1024 / 1024:.1f}MB). Maximum: {settings.max_upload_size_mb}MB",
            )
        chunks.append(chunk)

    content = b"".join(chunks)

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    log.info(
        "upload_validated",
        filename=file.filename,
        size_bytes=len(content),
        content_type=content_type,
    )

    return content
