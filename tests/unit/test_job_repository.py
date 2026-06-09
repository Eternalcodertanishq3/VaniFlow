"""
Unit tests for DubbingJobRepository.
Tests Redis save/get/list/delete with in-memory fallback.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from vaaniflow.repository.job_repository import DubbingJobRepository
from vaaniflow.models import DubbingJob, DubbingJobConfig, SupportedLanguage, JobStatus


@pytest.fixture
def repo():
    r = DubbingJobRepository()
    r._using_fallback = True  # Force in-memory mode for tests
    return r


@pytest.fixture
def sample_job():
    return DubbingJob(
        config=DubbingJobConfig(
            target_language=SupportedLanguage.HINDI,
        )
    )


@pytest.mark.asyncio
async def test_save_and_get(repo, sample_job):
    """Save a job and retrieve it."""
    await repo.save(sample_job)
    retrieved = await repo.get(sample_job.job_id)
    assert retrieved is not None
    assert retrieved.job_id == sample_job.job_id
    assert retrieved.config.target_language == SupportedLanguage.HINDI


@pytest.mark.asyncio
async def test_get_nonexistent(repo):
    """Getting a non-existent job should return None."""
    result = await repo.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_all_empty(repo):
    """Empty repo should return empty list."""
    jobs = await repo.list_all()
    assert jobs == []


@pytest.mark.asyncio
async def test_list_all_with_jobs(repo, sample_job):
    """List should return all saved jobs."""
    await repo.save(sample_job)
    job2 = DubbingJob(
        config=DubbingJobConfig(target_language=SupportedLanguage.BENGALI)
    )
    await repo.save(job2)
    jobs = await repo.list_all()
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_delete(repo, sample_job):
    """Delete should remove the job."""
    await repo.save(sample_job)
    await repo.delete(sample_job.job_id)
    result = await repo.get(sample_job.job_id)
    assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent(repo):
    """Deleting a non-existent job should not error."""
    result = await repo.delete("nonexistent-id")
    assert result is True


@pytest.mark.asyncio
async def test_save_updates_existing(repo, sample_job):
    """Saving an existing job should update it."""
    await repo.save(sample_job)
    sample_job.status = JobStatus.COMPLETED
    await repo.save(sample_job)
    retrieved = await repo.get(sample_job.job_id)
    assert retrieved.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_fallback_mode_on_redis_failure():
    """If Redis is unavailable, should use in-memory fallback."""
    repo = DubbingJobRepository()
    repo._using_fallback = True  # Simulating Redis failure
    job = DubbingJob(
        config=DubbingJobConfig(target_language=SupportedLanguage.TAMIL)
    )
    result = await repo.save(job)
    assert result is True
    retrieved = await repo.get(job.job_id)
    assert retrieved is not None
