"""Tests for retrieval services."""
import pytest
import numpy as np

from noteviz.core.retrieval.base import RetrievalConfig
from noteviz.core.retrieval.cosine import CosineRetrieval


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    texts = [
        "This is a test document about cats.",
        "Dogs are great pets.",
        "Birds can fly in the sky.",
        "Fish swim in the ocean."
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0]
    ]
    return texts, embeddings


@pytest.fixture
def retrieval_config():
    """Create a test retrieval configuration."""
    return RetrievalConfig(
        similarity_threshold=0.5,
        max_results=2
    )


@pytest.fixture
def retrieval_service(retrieval_config):
    """Create a test retrieval service."""
    return CosineRetrieval(retrieval_config)


def test_find_relevant_chunks(retrieval_service, sample_data):
    """Test finding relevant chunks."""
    texts, embeddings = sample_data
    
    # Index the data
    retrieval_service.index(texts, embeddings)
    
    # Test with a query similar to the first document
    query_embedding = [0.9, 0.1, 0.0]
    results = retrieval_service.find_relevant_chunks(query_embedding)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(result, tuple) for result in results)
    assert all(isinstance(text, str) and isinstance(score, float) 
              for text, score in results)
    assert all(0 <= score <= 1 for _, score in results)


def test_find_relevant_chunks_empty_input(retrieval_service):
    """Test handling of empty input."""
    with pytest.raises(ValueError) as exc_info:
        retrieval_service.find_relevant_chunks([0.5, 0.5, 0.0])
    assert "No indexed texts available" in str(exc_info.value)


def test_find_relevant_chunks_mismatched_input(retrieval_service):
    """Test handling of mismatched input lengths."""
    texts = ["Test 1", "Test 2"]
    embeddings = [[0.5, 0.5, 0.0]]  # One less embedding than text
    
    with pytest.raises(ValueError) as exc_info:
        retrieval_service.index(texts, embeddings)
    assert "Number of texts and embeddings must match" in str(exc_info.value)


def test_find_relevant_chunks_invalid_embeddings(retrieval_service, sample_data):
    """Test handling of invalid embeddings."""
    texts, embeddings = sample_data
    retrieval_service.index(texts, embeddings)
    
    # Test with invalid query embedding dimensions
    with pytest.raises(ValueError):
        retrieval_service.find_relevant_chunks([0.5, 0.5])  # Wrong dimensions 