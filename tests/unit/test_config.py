"""Tests for configuration classes."""
import pytest
from dataclasses import dataclass
from typing import Optional

from noteviz.core.llm.config import SummarizerConfig, TopicExtractorConfig
from noteviz.core.embedding.base import EmbeddingConfig
from noteviz.core.retrieval.base import RetrievalConfig


def test_summarizer_config_validation():
    """Test SummarizerConfig validation."""
    # Test valid config
    config = SummarizerConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        max_summary_length=200
    )
    assert config.model_name == "gpt-3.5-turbo"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.max_summary_length == 200

    # Test invalid temperature
    with pytest.raises(ValueError):
        SummarizerConfig(
            model_name="gpt-3.5-turbo",
            temperature=1.5,  # Invalid temperature
            max_tokens=100,
            max_summary_length=200
        )

    # Test invalid max_tokens
    with pytest.raises(ValueError):
        SummarizerConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=0,  # Invalid max_tokens
            max_summary_length=200
        )

    # Test invalid max_summary_length
    with pytest.raises(ValueError):
        SummarizerConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            max_summary_length=0  # Invalid max_summary_length
        )


def test_topic_extractor_config_validation():
    """Test TopicExtractorConfig validation."""
    # Test valid config
    config = TopicExtractorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        num_topics=3,
        max_context_chunks=1000
    )
    assert config.model_name == "gpt-3.5-turbo"
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.num_topics == 3
    assert config.max_context_chunks == 1000

    # Test invalid temperature
    with pytest.raises(ValueError):
        TopicExtractorConfig(
            model_name="gpt-3.5-turbo",
            temperature=1.5,  # Invalid temperature
            max_tokens=100,
            num_topics=3,
            max_context_chunks=1000
        )

    # Test invalid max_tokens
    with pytest.raises(ValueError):
        TopicExtractorConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=0,  # Invalid max_tokens
            num_topics=3,
            max_context_chunks=1000
        )

    # Test invalid num_topics
    with pytest.raises(ValueError):
        TopicExtractorConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            num_topics=0,  # Invalid num_topics
            max_context_chunks=1000
        )

    # Test invalid max_context_chunks
    with pytest.raises(ValueError):
        TopicExtractorConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            num_topics=3,
            max_context_chunks=0  # Invalid max_context_chunks
        )


def test_embedding_config_validation():
    """Test EmbeddingConfig validation."""
    # Test valid config
    config = EmbeddingConfig(
        model_name="text-embedding-3-small",
        device="cpu",
        batch_size=2
    )
    assert config.model_name == "text-embedding-3-small"
    assert config.device == "cpu"
    assert config.batch_size == 2

    # Test invalid device
    with pytest.raises(ValueError):
        EmbeddingConfig(
            model_name="text-embedding-3-small",
            device="invalid",  # Invalid device
            batch_size=2
        )

    # Test invalid batch_size
    with pytest.raises(ValueError):
        EmbeddingConfig(
            model_name="text-embedding-3-small",
            device="cpu",
            batch_size=0  # Invalid batch_size
        )


def test_retrieval_config_validation():
    """Test RetrievalConfig validation."""
    # Test valid config
    config = RetrievalConfig(
        similarity_threshold=0.7,
        max_results=3
    )
    assert config.similarity_threshold == 0.7
    assert config.max_results == 3

    # Test invalid similarity_threshold
    with pytest.raises(ValueError):
        RetrievalConfig(
            similarity_threshold=1.5,  # Invalid threshold
            max_results=3
        )

    # Test invalid max_results
    with pytest.raises(ValueError):
        RetrievalConfig(
            similarity_threshold=0.7,
            max_results=0  # Invalid max_results
        ) 