"""
Unit tests for the embedding service.
"""
import pytest
from unittest.mock import AsyncMock, patch
from openai import APIError, RateLimitError, APIStatusError

from noteviz.core.embedding.base import EmbeddingConfig
from noteviz.core.embedding.openai import OpenAIEmbeddingService


@pytest.fixture
def embedding_config():
    """Create a test embedding configuration."""
    return EmbeddingConfig(
        model_name="text-embedding-3-small",
        device="cpu",
        batch_size=2
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = AsyncMock(
        data=[AsyncMock(embedding=[0.1, 0.2, 0.3])]
    )
    return mock_client


@pytest.fixture
def embedding_service(embedding_config, mock_openai_client):
    """Create a test embedding service."""
    return OpenAIEmbeddingService(embedding_config, client=mock_openai_client)


@pytest.mark.asyncio
async def test_generate_embeddings(embedding_service):
    """Test embedding generation."""
    texts = ["This is a test sentence."]
    embeddings = await embedding_service.generate_embeddings(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)


@pytest.mark.asyncio
async def test_get_model_info(embedding_service):
    """Test model information retrieval."""
    info = await embedding_service.get_model_info()
    
    assert isinstance(info, dict)
    assert "provider" in info
    assert "model" in info
    assert "dimensions" in info


@pytest.mark.asyncio
async def test_generate_embeddings_api_error(embedding_service, mock_openai_client):
    """Test handling of API errors during embedding generation."""
    texts = ["This is a test sentence."]
    
    # Mock OpenAI API error
    mock_request = AsyncMock()
    mock_body = {"error": {"message": "API Error"}}
    mock_openai_client.embeddings.create.side_effect = APIError(
        "API Error",
        request=mock_request,
        body=mock_body
    )
    
    with pytest.raises(APIError):
        await embedding_service.generate_embeddings(texts)


@pytest.mark.asyncio
async def test_generate_embeddings_rate_limit(embedding_service, mock_openai_client):
    """Test handling of rate limit errors during embedding generation."""
    texts = ["This is a test sentence."]
    
    # Mock OpenAI rate limit error
    mock_response = AsyncMock(status_code=429)
    mock_body = {"error": {"message": "Rate limit exceeded"}}
    mock_openai_client.embeddings.create.side_effect = APIStatusError(
        "Rate limit exceeded",
        response=mock_response,
        body=mock_body
    )
    
    with pytest.raises(APIStatusError):
        await embedding_service.generate_embeddings(texts)


@pytest.mark.asyncio
async def test_generate_embeddings_empty_input(embedding_service):
    """Test handling of empty input during embedding generation."""
    texts = []
    
    with pytest.raises(ValueError) as exc_info:
        await embedding_service.generate_embeddings(texts)
    assert "No texts provided for embedding generation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_embeddings_invalid_response(embedding_service, mock_openai_client):
    """Test handling of invalid API response during embedding generation."""
    texts = ["This is a test sentence."]
    
    # Mock invalid response (missing embedding field)
    mock_data = AsyncMock()
    delattr(mock_data, "embedding")  # Remove the embedding attribute
    mock_response = AsyncMock()
    mock_response.data = [mock_data]
    mock_openai_client.embeddings.create.return_value = mock_response
    
    with pytest.raises(AttributeError):
        await embedding_service.generate_embeddings(texts) 