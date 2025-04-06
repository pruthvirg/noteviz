"""
Embedding module for NoteViz.
"""
from .base import EmbeddingService, EmbeddingConfig
from .openai import OpenAIEmbeddingService

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig",
    "OpenAIEmbeddingService",
]
