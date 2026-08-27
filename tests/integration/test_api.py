"""
Integration tests for the FastAPI API.
Tests endpoints with mocked pipeline.
"""
from unittest.mock import AsyncMock, patch


def test_health_endpoint(api_client):
    """Health endpoint should return 200 with status."""
    response = api_client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_get_nonexistent_job(api_client):
    """Getting a nonexistent job should return 404."""
    response = api_client.get("/jobs/nonexistent-uuid")
    assert response.status_code == 404


def test_create_job(api_client):
    """Creating a job should return 202 with job_id."""
    import io

    fake_audio = io.BytesIO(b"fake_audio_content" * 100)

    with patch("api.routes.jobs.run_pipeline_task", new_callable=AsyncMock):
        response = api_client.post(
            "/jobs/",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={
                "target_language": "hi",
                "source_language": "en",
                "tts_provider": "gtts",
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["progress_pct"] == 0.0


def test_create_and_get_job(api_client):
    """Create a job then retrieve its status."""
    import io

    fake_audio = io.BytesIO(b"fake_audio_content" * 100)

    with patch("api.routes.jobs.run_pipeline_task", new_callable=AsyncMock):
        create_response = api_client.post(
            "/jobs/",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={
                "target_language": "ta",
                "source_language": "en",
            },
        )

    job_id = create_response.json()["job_id"]

    get_response = api_client.get(f"/jobs/{job_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "pending"


def test_download_before_completion(api_client):
    """Downloading before job completes should return 400."""
    import io

    fake_audio = io.BytesIO(b"fake_audio_content" * 100)

    with patch("api.routes.jobs.run_pipeline_task", new_callable=AsyncMock):
        create_response = api_client.post(
            "/jobs/",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={
                "target_language": "bn",
                "source_language": "en",
            },
        )

    job_id = create_response.json()["job_id"]

    download_response = api_client.get(f"/jobs/{job_id}/download")
    assert download_response.status_code == 400


def test_list_jobs(api_client):
    """List jobs endpoint should return a list."""
    response = api_client.get("/jobs/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
