"""Tests for the LLM services."""
import json
import os
import pytest
from unittest.mock import AsyncMock, patch

from openai import APIError, RateLimitError, APIStatusError, AuthenticationError
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice

from noteviz.core.llm.config import SummarizerConfig, TopicExtractorConfig
from noteviz.core.llm.base import Topic
from noteviz.core.llm.openai import OpenAISummarizer, OpenAITopicExtractor


@pytest.fixture(autouse=True)
def mock_api_key():
    """Mock OpenAI API key for all tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def summarizer_config():
    """Create a test summarizer configuration."""
    return SummarizerConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        max_summary_length=100
    )


@pytest.fixture
def topic_extractor_config():
    """Create a test topic extractor configuration."""
    return TopicExtractorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500,
        num_topics=3,
        max_context_chunks=5
    )


@pytest.mark.asyncio
async def test_summarize(summarizer_config):
    """Test text summarization."""
    with patch("noteviz.core.llm.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="Test summary"))]
        )
        mock_openai.return_value = mock_client

        service = OpenAISummarizer(summarizer_config)
        summary = await service.summarize("Test text")
        assert summary == "Test summary"


@pytest.mark.asyncio
async def test_extract_topics(topic_extractor_config):
    """Test topic extraction."""
    with patch("noteviz.core.llm.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_response = json.dumps([
            {
                "name": "Topic 1",
                "description": "Description 1",
                "confidence": 0.8,
                "keywords": ["key1", "key2"]
            },
            {
                "name": "Topic 2",
                "description": "Description 2",
                "confidence": 0.7,
                "keywords": ["key3", "key4"]
            },
            {
                "name": "Topic 3",
                "description": "Description 3",
                "confidence": 0.6,
                "keywords": ["key5", "key6"]
            }
        ])
        mock_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content=mock_response))]
        )
        mock_openai.return_value = mock_client

        service = OpenAITopicExtractor(topic_extractor_config)
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5", "Chunk 6"]
        topics = await service.extract_topics(chunks)
        assert len(topics) == 3
        assert topics[0].name == "Topic 1"
        assert topics[0].confidence == 0.8


@pytest.mark.asyncio
async def test_summarize_api_error(summarizer_config):
    """Test handling of API errors during summarization."""
    with patch("noteviz.core.llm.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_request = AsyncMock()
        mock_body = {"error": {"message": "API Error"}}
        mock_client.chat.completions.create.side_effect = APIError(
            "API Error",
            request=mock_request,
            body=mock_body
        )
        mock_openai.return_value = mock_client

        service = OpenAISummarizer(summarizer_config)
        with pytest.raises(APIError):
            await service.summarize("Test text")


@pytest.mark.asyncio
async def test_summarize_rate_limit(summarizer_config):
    """Test handling of rate limit errors during summarization."""
    with patch("noteviz.core.llm.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_response = AsyncMock(status_code=429)
        mock_body = {"error": {"message": "Rate limit exceeded"}}
        mock_client.chat.completions.create.side_effect = APIStatusError(
            "Rate limit exceeded",
            response=mock_response,
            body=mock_body
        )
        mock_openai.return_value = mock_client

        service = OpenAISummarizer(summarizer_config)
        with pytest.raises(APIStatusError):
            await service.summarize("Test text")


@pytest.mark.asyncio
async def test_extract_topics_invalid_input(topic_extractor_config):
    """Test handling of invalid input during topic extraction."""
    with patch("noteviz.core.llm.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="invalid json"))]
        )
        mock_openai.return_value = mock_client

        service = OpenAITopicExtractor(topic_extractor_config)
        chunks = ["Chunk 1"]

        with pytest.raises(json.JSONDecodeError):
            await service.extract_topics(chunks)


@pytest.mark.asyncio
async def test_extract_topics_empty_input(topic_extractor_config):
    """Test handling of empty input during topic extraction."""
    service = OpenAITopicExtractor(topic_extractor_config)
    chunks = []

    with pytest.raises(ValueError) as exc_info:
        await service.extract_topics(chunks)
    assert "No text chunks provided" in str(exc_info.value) 