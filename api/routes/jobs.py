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
    DubbingJobResponse, JobStatus, SupportedLanguage, TTSProvider,
)
from vaaniflow.pipeline import VaaniFlowPipeline
from vaaniflow.config import settings
from vaaniflow.repository.job_repository import DubbingJobRepository

router = APIRouter()
log = structlog.get_logger(__name__)
pipeline = VaaniFlowPipeline()
job_repo = DubbingJobRepository()


@router.post("/", response_model=DubbingJobResponse, status_code=202)
async def create_dubbing_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: SupportedLanguage = Form(...),
    source_language: SupportedLanguage = Form(default=SupportedLanguage.ENGLISH),
    tts_provider: TTSProvider = Form(default=TTSProvider.SARVAM),
    voice_id: str | None = Form(default=None),
):
    """
    Create a new dubbing job.
    Accepts audio/video file + target language config.
    Returns job_id immediately; processing happens in background.
    """
    # Build config from form data
    config = DubbingJobConfig(
        source_language=source_language,
        target_language=target_language,
        tts_provider=tts_provider,
        voice_id=voice_id,
    )
    job = DubbingJob(config=config)
    await job_repo.save(job)

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    content = await file.read()
    
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
    except Exception as e:
        log.error("background_task_failed", job_id=job.job_id, error=str(e))
    finally:
        input_path.unlink(missing_ok=True)
