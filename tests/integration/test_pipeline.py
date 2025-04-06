"""Integration tests for the full pipeline."""
import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
from reportlab.pdfgen import canvas

from noteviz.core.pdf import PDFConfig, PyPDFProcessor
from noteviz.core.embedding import EmbeddingConfig, OpenAIEmbeddingService
from noteviz.core.llm import (
    SummarizerConfig,
    TopicExtractorConfig,
    OpenAILLMService,
    Topic
)
from noteviz.core.retrieval import RetrievalConfig, CosineRetrieval


@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a test PDF file."""
    pdf_path = tmp_path / "test.pdf"
    # Create a simple test PDF with some content
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 700, "This is a test document for integration testing.")
    c.drawString(100, 650, "It contains multiple paragraphs of text.")
    c.drawString(100, 600, "The text will be processed by our pipeline.")
    c.save()
    return pdf_path


@pytest.fixture
def pdf_config():
    """Create a test PDF configuration."""
    return PDFConfig(
        chunk_size=100,
        chunk_overlap=20
    )


@pytest.fixture
def embedding_config():
    """Create a test embedding configuration."""
    return EmbeddingConfig(
        model_name="text-embedding-3-small",
        device="cpu",
        batch_size=2
    )


@pytest.fixture
def summarizer_config():
    """Create a test summarizer configuration."""
    return SummarizerConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )


@pytest.fixture
def topic_config():
    """Create a test topic extractor configuration."""
    return TopicExtractorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        num_topics=3
    )


@pytest.fixture
def retrieval_config():
    """Create a test retrieval configuration."""
    return RetrievalConfig(
        similarity_threshold=0.7,
        max_results=5
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    
    # Mock chat completion responses
    client.chat.completions.create.return_value = AsyncMock(
        choices=[
            AsyncMock(
                message=AsyncMock(
                    content=json.dumps([
                        {
                            "name": "Test Topic",
                            "description": "A test topic",
                            "confidence": 0.8,
                            "keywords": ["test", "topic", "example"]
                        }
                    ])
                )
            )
        ]
    )
    
    # Mock embedding responses
    client.embeddings.create.return_value = AsyncMock(
        data=[
            AsyncMock(embedding=[0.1] * 1536),
            AsyncMock(embedding=[0.2] * 1536)
        ]
    )
    
    return client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline(
    test_pdf_path,
    pdf_config,
    embedding_config,
    summarizer_config,
    topic_config,
    retrieval_config,
    mock_openai_client
):
    """Test the full pipeline from PDF to topics and summary."""
    # Initialize services
    pdf_processor = PyPDFProcessor(pdf_config)
    embedding_service = OpenAIEmbeddingService(embedding_config, client=mock_openai_client)
    llm_service = OpenAILLMService(summarizer_config, topic_config, client=mock_openai_client)
    retrieval_service = CosineRetrieval(retrieval_config)
    
    # Process PDF
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    assert len(chunks) > 0
    
    # Generate embeddings
    embeddings = await embedding_service.generate_embeddings(chunks)
    assert len(embeddings) == len(chunks)
    
    # Extract topics
    text = "\n".join(chunks)
    topics = await llm_service.extract_topics(text)
    assert len(topics) > 0
    assert all(isinstance(topic, Topic) for topic in topics)
    
    # Generate summary
    summary = await llm_service.generate_summary(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Identify key concepts
    key_concepts = await llm_service.identify_key_concepts(text)
    assert isinstance(key_concepts, list)
    assert len(key_concepts) > 0 