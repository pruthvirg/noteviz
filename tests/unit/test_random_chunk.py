"""
Unit tests for the random chunk topic extractor.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from noteviz.core.llm.config import TopicExtractorConfig
from noteviz.core.llm.random_chunk import RandomChunkTopicExtractor


@pytest.fixture
def topic_config():
    """Create a test configuration for the topic extractor."""
    return TopicExtractorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        num_topics=5
    )


@pytest.fixture
def mock_openai():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def topic_extractor(topic_config, mock_openai):
    """Create a topic extractor instance."""
    return RandomChunkTopicExtractor(topic_config, client=mock_openai)


@pytest.fixture
def sample_chunks():
    """Create sample text chunks for testing."""
    return [
        "This is the first chunk of text.",
        "This is the second chunk of text.",
        "This is the third chunk of text.",
        "This is the fourth chunk of text."
    ]


@pytest.mark.asyncio
async def test_extract_topics(topic_extractor, mock_openai, sample_chunks):
    """Test topic extraction with random chunks."""
    # Set up mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps([
                    {
                        "name": "Topic 1",
                        "description": "Description 1",
                        "confidence": 0.8,
                        "keywords": ["key1", "key2"]
                    },
                    {
                        "name": "Topic 2",
                        "description": "Description 2",
                        "confidence": 0.9,
                        "keywords": ["key3", "key4"]
                    }
                ])
            )
        )
    ]
    mock_openai.chat.completions.create.return_value = mock_response

    # Extract topics
    topics = await topic_extractor.extract_topics(sample_chunks)

    # Verify OpenAI was called correctly
    mock_openai.chat.completions.create.assert_called_once()
    call_args = mock_openai.chat.completions.create.call_args[1]
    assert call_args["model"] == topic_extractor.config.model_name
    assert call_args["temperature"] == topic_extractor.config.temperature
    assert call_args["max_tokens"] == topic_extractor.config.max_tokens

    # Verify the topics
    assert len(topics) == 2
    assert topics[0].name == "Topic 1"
    assert topics[0].description == "Description 1"
    assert topics[0].confidence == 0.8
    assert topics[0].keywords == ["key1", "key2"]
    assert topics[1].name == "Topic 2"
    assert topics[1].description == "Description 2"
    assert topics[1].confidence == 0.9
    assert topics[1].keywords == ["key3", "key4"]


@pytest.mark.asyncio
async def test_extract_topics_empty_chunks(topic_extractor):
    """Test topic extraction with empty chunks."""
    with pytest.raises(ValueError, match="No text chunks provided"):
        await topic_extractor.extract_topics([])


@pytest.mark.asyncio
async def test_extract_topics_api_error(topic_extractor, mock_openai, sample_chunks):
    """Test handling of API errors."""
    # Make the mock raise an exception
    mock_openai.chat.completions.create.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        await topic_extractor.extract_topics(sample_chunks)


@pytest.mark.asyncio
async def test_extract_topics_invalid_json(topic_extractor, mock_openai, sample_chunks):
    """Test handling of invalid JSON response."""
    # Mock OpenAI to return invalid JSON
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Invalid JSON"
            )
        )
    ]
    mock_openai.chat.completions.create.return_value = mock_response

    with pytest.raises(json.JSONDecodeError):
        await topic_extractor.extract_topics(sample_chunks)


@pytest.mark.asyncio
async def test_extract_topics_random_selection(topic_extractor, mock_openai, sample_chunks):
    """Test that chunks are randomly selected when more than 3 are provided."""
    # Set up mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps([
                    {
                        "name": "Topic 1",
                        "description": "Description 1",
                        "confidence": 0.8,
                        "keywords": ["key1", "key2"]
                    }
                ])
            )
        )
    ]
    mock_openai.chat.completions.create.return_value = mock_response

    # Extract topics
    await topic_extractor.extract_topics(sample_chunks)

    # Get the chunks used in the prompt
    call_args = mock_openai.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    prompt = messages[0]["content"]

    # Count the number of chunks in the prompt
    chunk_count = prompt.count("This is the")
    assert chunk_count == 3  # Should only use 3 chunks 