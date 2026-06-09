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

router = APIRouter()
log = structlog.get_logger(__name__)
pipeline = VaaniFlowPipeline()
jobs_store: dict[str, DubbingJob] = {}  # In production: Redis


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
    jobs_store[job.job_id] = job

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

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
    job = jobs_store.get(job_id)
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
    job = jobs_store.get(job_id)
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
        for job in jobs_store.values()
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
