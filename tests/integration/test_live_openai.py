"""
Integration tests for OpenAI services using a real API key.
"""
import os
import pytest
import asyncio
from pathlib import Path
from reportlab.pdfgen import canvas

from noteviz.core.pdf import PDFConfig, PyPDFProcessor
from noteviz.core.embedding import EmbeddingConfig, OpenAIEmbeddingService
from noteviz.core.llm import (
    SummarizerConfig,
    TopicExtractorConfig,
    OpenAILLMService,
)
from noteviz.core.retrieval import RetrievalConfig, CosineRetrieval


@pytest.fixture
def test_pdf_path(tmp_path):
    """Create a test PDF file."""
    pdf_path = tmp_path / "test.pdf"
    # Create a simple test PDF with some content
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 700, "This is a test document for OpenAI integration testing.")
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


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_openai_integration(
    test_pdf_path,
    pdf_config,
    embedding_config,
    summarizer_config,
    topic_config,
    retrieval_config
):
    """Test the full pipeline with a real OpenAI API key."""
    # Skip if no API key is set
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set in environment")
    
    # Initialize services
    pdf_processor = PyPDFProcessor(pdf_config)
    embedding_service = OpenAIEmbeddingService(embedding_config)
    llm_service = OpenAILLMService(summarizer_config, topic_config)
    retrieval_service = CosineRetrieval(retrieval_config)
    
    # Process PDF
    print(f"Processing PDF: {test_pdf_path}")
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    assert len(chunks) > 0
    print(f"Extracted {len(chunks)} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = await embedding_service.generate_embeddings(chunks)
    assert len(embeddings) == len(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Extract topics
    print("\nExtracting topics...")
    text = "\n".join(chunks)
    topics = await llm_service.extract_topics(text)
    assert len(topics) > 0
    print("\nTopics:")
    for topic in topics:
        print(f"- {topic.name}: {topic.description}")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = await llm_service.generate_summary(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
    print(f"\nSummary:\n{summary}")
    
    # Identify key concepts
    print("\nIdentifying key concepts...")
    key_concepts = await llm_service.identify_key_concepts(text)
    assert isinstance(key_concepts, list)
    assert len(key_concepts) > 0
    print("\nKey Concepts:")
    for concept in key_concepts:
        print(f"- {concept}")


if __name__ == "__main__":
    # This allows running the test directly with python
    asyncio.run(pytest.main([__file__, "-v", "-s"])) 