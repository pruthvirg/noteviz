"""Tests for the OpenAI flowchart generator."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from noteviz.core.flowchart.openai import OpenAIFlowchartGenerator
from noteviz.core.flowchart.base import Flowchart, Node, Edge
from noteviz.core.llm.base import Topic


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("noteviz.core.flowchart.openai.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_topic_extractor():
    """Create a mock topic extractor."""
    with patch("noteviz.core.flowchart.openai.OpenAITopicExtractor") as mock:
        mock_instance = AsyncMock()
        mock_instance.extract_topics.return_value = [
            Topic(
                name="Test Topic",
                description="Test Description",
                confidence=0.9,
                keywords=["keyword1", "keyword2"]
            )
        ]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_generate_flowchart_with_provided_topic_and_keywords(mock_openai_client):
    """Test flowchart generation with provided topic and keywords."""
    # Mock OpenAI response
    mock_response = {
        "nodes": [
            {
                "id": "node1",
                "label": "Concept 1",
                "description": "Description 1",
                "confidence": 0.9
            },
            {
                "id": "node2",
                "label": "Concept 2",
                "description": "Description 2",
                "confidence": 0.8
            }
        ],
        "edges": [
            {
                "source": "node1",
                "target": "node2",
                "label": "relates to",
                "description": "Relationship description"
            }
        ]
    }
    mock_openai_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content=json.dumps(mock_response)))]
    )

    generator = OpenAIFlowchartGenerator(client=mock_openai_client)
    flowchart = await generator.generate_flowchart(
        text="Sample text",
        topic="Test Topic",
        keywords=["keyword1", "keyword2"]
    )

    assert isinstance(flowchart, Flowchart)
    assert len(flowchart.nodes) == 2
    assert len(flowchart.edges) == 1
    assert flowchart.title == "Flowchart: Test Topic"


@pytest.mark.asyncio
async def test_generate_flowchart_with_extracted_topic_and_keywords(
    mock_openai_client,
    mock_topic_extractor
):
    """Test flowchart generation with extracted topic and keywords."""
    # Mock OpenAI response
    mock_response = {
        "nodes": [
            {
                "id": "node1",
                "label": "Concept 1",
                "description": "Description 1",
                "confidence": 0.9
            }
        ],
        "edges": []
    }
    mock_openai_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content=json.dumps(mock_response)))]
    )

    generator = OpenAIFlowchartGenerator(client=mock_openai_client)
    flowchart = await generator.generate_flowchart(text="Sample text")

    assert isinstance(flowchart, Flowchart)
    assert len(flowchart.nodes) == 1
    assert len(flowchart.edges) == 0
    assert flowchart.title == "Flowchart: Test Topic"
    mock_topic_extractor.extract_topics.assert_called_once()


@pytest.mark.asyncio
async def test_generate_flowchart_invalid_response(mock_openai_client):
    """Test handling of invalid OpenAI response."""
    mock_openai_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content="invalid json"))]
    )

    generator = OpenAIFlowchartGenerator(client=mock_openai_client)
    with pytest.raises(json.JSONDecodeError):
        await generator.generate_flowchart(
            text="Sample text",
            topic="Test Topic",
            keywords=["keyword1"]
        )


@pytest.mark.asyncio
async def test_generate_flowchart_missing_required_fields(mock_openai_client):
    """Test handling of response missing required fields."""
    mock_response = {
        "nodes": [
            {
                "id": "node1",
                # Missing required fields
            }
        ],
        "edges": []
    }
    mock_openai_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content=json.dumps(mock_response)))]
    )

    generator = OpenAIFlowchartGenerator(client=mock_openai_client)
    with pytest.raises(KeyError):
        await generator.generate_flowchart(
            text="Sample text",
            topic="Test Topic",
            keywords=["keyword1"]
        ) 