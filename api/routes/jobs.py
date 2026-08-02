"""
Dubbing job API endpoints.
"""
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
import structlog
import tempfile
from pathlib import Path

from vaaniflow.models import (
    DubbingJob, DubbingJobConfig, DubbingJobRequest,
    DubbingJobResponse, JobStatus, SupportedLanguage, TTSProvider, TranslationProvider,
)
from vaaniflow.pipeline import VaaniFlowPipeline
from vaaniflow.config import settings
from vaaniflow.repository.job_repository import DubbingJobRepository
from api.middleware.upload_validation import validate_upload

router = APIRouter()
log = structlog.get_logger(__name__)
pipeline = VaaniFlowPipeline()
job_repo = DubbingJobRepository()


@router.post("/", response_model=DubbingJobResponse, status_code=202)
async def create_dubbing_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: SupportedLanguage = Form(...),
    source_language: SupportedLanguage = Form(default=SupportedLanguage.AUTO),
    tts_provider: TTSProvider = Form(default=TTSProvider.SARVAM),
    translation_provider: TranslationProvider | None = Form(default=None),
    voice_id: str | None = Form(default=None),
    speaker_gender: str = Form(default="Male"),
    translation_mode: str = Form(default="formal"),
    loudness: float = Form(default=1.5, ge=0.5, le=3.0),
    preserve_ambient: bool = Form(default=True),
):
    """
    Create a new dubbing job.
    Accepts audio/video file + target language config.
    Returns job_id immediately; processing happens in background.
    """
    # Validate uploaded file (size, format, content-type)
    content = await validate_upload(file)

    chosen_translation_provider = translation_provider
    if chosen_translation_provider is None:
        chosen_translation_provider = (
            TranslationProvider.SARVAM
            if tts_provider == TTSProvider.SARVAM
            else TranslationProvider.GOOGLE
        )

    # Build config from form data
    config = DubbingJobConfig(
        source_language=source_language,
        target_language=target_language,
        tts_provider=tts_provider,
        translation_provider=chosen_translation_provider,
        voice_id=voice_id,
        speaker_gender=speaker_gender,
        translation_mode=translation_mode,
        loudness=loudness,
        preserve_ambient=preserve_ambient,
    )
    job = DubbingJob(config=config)
    await job_repo.save(job)

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    def write_temp_file(data: bytes, suf: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(data)
            return Path(tmp.name)

    tmp_path = await asyncio.to_thread(write_temp_file, content, suffix)

    log.info(
        "job_created",
        job_id=job.job_id,
        target_lang=target_language,
        filename=file.filename,
        size_bytes=len(content),
        speaker_gender=speaker_gender,
        translation_mode=translation_mode,
    )

    background_tasks.add_task(run_pipeline_task, job, tmp_path)

    return DubbingJobResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
    )


@router.get("/{job_id}", response_model=DubbingJobResponse)
async def get_job_status(job_id: str):
    """Get current status and progress of a dubbing job."""
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return DubbingJobResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        output_url=f"/jobs/{job_id}/download" if job.status == JobStatus.COMPLETED else None,
        error=job.error_message,
    )


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running or pending dubbing job.
    Sets status to FAILED with cancellation message.
    """
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check if job is in a cancellable state
    terminal_statuses = {JobStatus.COMPLETED, JobStatus.FAILED}
    if job.status in terminal_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Job already finished with status: {job.status}",
        )

    previous_status = job.status
    job.status = JobStatus.FAILED
    job.error_message = "Cancelled by user"
    await job_repo.save(job)

    log.info("job_cancelled", job_id=job_id, previous_status=previous_status)

    return {
        "job_id": job_id,
        "status": job.status,
        "message": "Job cancelled successfully",
    }


@router.get("/{job_id}/download")
async def download_result(job_id: str):
    """Download completed dubbed audio file."""
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        job.output_path,
        media_type="audio/wav",
        filename=f"dubbed_{job_id}.wav",
    )


@router.get("/{job_id}/subtitles/{format}")
async def download_subtitles(job_id: str, format: str):
    """Download SRT or VTT subtitles. format: 'srt' or 'vtt'"""
    if format not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="Format must be 'srt' or 'vtt'")
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    subtitle_path = Path(settings.output_dir) / f"{job_id}.{format}"
    if not subtitle_path.exists():
        raise HTTPException(status_code=404, detail=f"{format.upper()} file not found")

    media_type = "text/srt" if format == "srt" else "text/vtt"
    return FileResponse(subtitle_path, media_type=media_type, filename=f"{job_id}.{format}")


@router.get("/", response_model=list[DubbingJobResponse])
async def list_jobs():
    """List all dubbing jobs."""
    return [
        DubbingJobResponse(
            job_id=job.job_id,
            status=job.status,
            progress_pct=job.progress_pct,
            output_url=f"/jobs/{job.job_id}/download" if job.status == JobStatus.COMPLETED else None,
            error=job.error_message,
        )
        for job in await job_repo.list_all()
    ]


async def run_pipeline_task(job: DubbingJob, input_path: Path):
    """Background task to run the pipeline."""
    try:
        output_path = await pipeline.run(job, input_path)
        job.output_path = str(output_path)
        await job_repo.save(job)
    except Exception as e:
        log.error("background_task_failed", job_id=job.job_id, error=str(e))
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        await job_repo.save(job)
    finally:
        input_path.unlink(missing_ok=True)

