"""Base classes for retrieval services."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RetrievalConfig:
    """Configuration for retrieval services."""
    
    similarity_threshold: float = 0.7
    """Minimum similarity score for a chunk to be considered relevant."""
    
    max_results: int = 5
    """Maximum number of results to return."""
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_results <= 0:
            raise ValueError("max_results must be greater than 0")


class RetrievalService(ABC):
    """Base class for retrieval services."""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
    
    @abstractmethod
    def index(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Index the texts and their embeddings.
        
        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors.
        """
        pass
    
    @abstractmethod
    def find_relevant_chunks(self, query_embedding: List[float]) -> List[Tuple[str, float]]:
        """Find relevant text chunks based on similarity.
        
        Args:
            query_embedding: Embedding vector of the query.
            
        Returns:
            List of tuples containing (text, similarity_score).
        """
        pass 