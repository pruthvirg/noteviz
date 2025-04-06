"""Configuration classes for LLM services."""
from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM service."""
    model_name: str = Field(default="gpt-3.5-turbo", description="Name of the model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for text generation")
    max_tokens: int = Field(default=500, gt=0, description="Maximum number of tokens to generate")


class SummarizerConfig(LLMConfig):
    """Configuration for text summarization."""
    max_summary_length: Optional[int] = Field(default=None, gt=0, description="Maximum length of the summary in words")


class TopicExtractorConfig(LLMConfig):
    """Configuration for topic extraction."""
    num_topics: int = Field(default=5, gt=0, description="Number of topics to extract")
    max_context_chunks: int = Field(default=1000, gt=0, description="Maximum number of context chunks to process") 