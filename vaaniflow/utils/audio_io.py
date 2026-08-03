"""
Async audio I/O utilities with memory guards.

Replaces blocking Path.read_bytes() / Path.write_bytes() in the pipeline
with non-blocking async I/O that runs in a thread pool, plus a configurable
size guard to reject files that would exceed server memory.
"""

import asyncio
from pathlib import Path

import structlog

from vaaniflow.config import settings
from vaaniflow.exceptions import AudioProcessingError

log = structlog.get_logger(__name__)


async def read_audio_async(path: Path) -> bytes:
    """
    Read an audio file asynchronously with memory guard.

    Uses asyncio.to_thread so the blocking file read happens in the
    default thread-pool executor, keeping the event loop free for
    concurrent TTS / translation calls.

    Raises:
        AudioProcessingError: If file exceeds max_audio_bytes setting.
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    file_size = path.stat().st_size
    max_bytes = getattr(settings, "max_audio_bytes", 500 * 1024 * 1024)

    if file_size > max_bytes:
        raise AudioProcessingError(
            f"Audio file too large ({file_size / 1024 / 1024:.1f}MB). "
            f"Maximum: {max_bytes / 1024 / 1024:.0f}MB. "
            f"Consider splitting into shorter segments."
        )

    data = await asyncio.to_thread(path.read_bytes)
    log.debug("audio_read_async", path=str(path), size_bytes=len(data))
    return data


async def write_audio_async(path: Path, data: bytes) -> None:
    """
    Write audio data to file asynchronously.

    Uses asyncio.to_thread to avoid blocking the event loop
    during large file writes (e.g. 50MB+ dubbed audio output).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, data)
    log.debug("audio_written_async", path=str(path), size_bytes=len(data))
