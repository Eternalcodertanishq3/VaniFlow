"""
Unit tests for API key authentication middleware.
"""
import pytest
from unittest.mock import patch
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from api.middleware.auth_middleware import APIKeyAuthMiddleware


# Simple test app
async def hello(request):
    return JSONResponse({"message": "ok"})


async def health(request):
    return JSONResponse({"status": "healthy"})


async def metrics(request):
    return JSONResponse({"metrics": "ok"})


def create_test_app():
    app = Starlette(
        routes=[
            Route("/api/test", hello),
            Route("/health/", health),
            Route("/health/ready", health),
            Route("/metrics", metrics),
        ]
    )
    app.add_middleware(APIKeyAuthMiddleware)
    return app


class TestAPIKeyAuth:
    """Tests for API key authentication middleware."""

    def test_auth_disabled_when_no_key(self):
        """When VAANIFLOW_API_KEY is empty, all requests pass through."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = ""
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/api/test")
            assert response.status_code == 200
            assert response.json()["message"] == "ok"

    def test_auth_rejects_missing_key(self):
        """When API key is configured, requests without key are rejected."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = "test-secret-key"
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/api/test")
            assert response.status_code == 401
            assert "Invalid or missing API key" in response.json()["detail"]

    def test_auth_rejects_wrong_key(self):
        """Wrong API key is rejected."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = "test-secret-key"
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/api/test", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401

    def test_auth_accepts_correct_key(self):
        """Correct API key passes through."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = "test-secret-key"
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/api/test", headers={"X-API-Key": "test-secret-key"})
            assert response.status_code == 200
            assert response.json()["message"] == "ok"

    def test_health_bypasses_auth(self):
        """Health endpoints should bypass auth even when key is configured."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = "test-secret-key"
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/health/")
            assert response.status_code == 200

            response = client.get("/health/ready")
            assert response.status_code == 200

    def test_metrics_bypasses_auth(self):
        """Metrics endpoint should bypass auth."""
        with patch("api.middleware.auth_middleware.settings") as mock_settings:
            mock_settings.api_key = "test-secret-key"
            app = create_test_app()
            client = TestClient(app)

            response = client.get("/metrics")
            assert response.status_code == 200
