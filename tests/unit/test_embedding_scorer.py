import pytest
from unittest.mock import MagicMock
from vaaniflow.quality.embedding_scorer import EmbeddingQualityScorer

@pytest.fixture
def scorer():
    return EmbeddingQualityScorer(threshold=0.75, enabled=True)

@pytest.mark.asyncio
async def test_disabled_returns_pass():
    s = EmbeddingQualityScorer(enabled=False)
    result = await s.score("Hello world", "Hello world")
    assert result.passed is True
    assert result.cosine_similarity == 1.0

@pytest.mark.asyncio
async def test_empty_text_fails(scorer):
    scorer._model = MagicMock()
    result = await scorer.score("", "something")
    assert result.passed is False

@pytest.mark.asyncio
async def test_model_unavailable_graceful_fallback(scorer):
    scorer._model_load_failed = True
    result = await scorer.score("Hello", "World")
    assert result.passed is True

@pytest.mark.asyncio
async def test_identical_text_high_similarity(scorer):
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scorer._model = mock_model
    result = await scorer.score("Hello how are you", "Hello how are you")
    assert result.cosine_similarity > 0.95
    assert result.passed is True

@pytest.mark.asyncio
async def test_dissimilar_text_low_similarity(scorer):
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    scorer._model = mock_model
    result = await scorer.score("Hello", "Completely unrelated topic")
    assert result.cosine_similarity < 0.6
