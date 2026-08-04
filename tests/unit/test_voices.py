"""
Unit tests for voices catalog API endpoint.
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_all_voices():
    """Test GET /voices/ returns voices grouped by provider."""
    response = client.get("/voices/")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert "sarvam" in data["voices"]
    assert "elevenlabs" in data["voices"]
    assert "gtts" in data["voices"]


def test_filter_voices_by_provider():
    """Test GET /voices/?provider=sarvam filters correctly."""
    response = client.get("/voices/?provider=sarvam")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "sarvam"
    assert data["count"] > 0
    for voice in data["voices"]:
        assert voice["id"] in {"arvind", "amartya", "ratan", "meera", "kavya", "anvita"}


def test_filter_voices_by_provider_and_gender():
    """Test GET /voices/?provider=sarvam&gender=Male returns only male voices."""
    response = client.get("/voices/?provider=sarvam&gender=Male")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "sarvam"
    for voice in data["voices"]:
        assert voice["gender"].lower() == "male"
