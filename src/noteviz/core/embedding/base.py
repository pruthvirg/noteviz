"""
Base interface for embedding services.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Base configuration for embedding services."""
    model_name: str = Field(..., description="Name of the embedding model to use")
    device: str = Field(default="cpu", pattern="^(cpu|cuda)$", description="Device to run the model on")
    batch_size: int = Field(default=32, gt=0, description="Batch size for processing")


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for.
            
        Returns:
            List of embedding vectors.
        """
        pass
    
    @abstractmethod
    async def get_model_info(self) -> dict:
        """Get information about the embedding model.
        
        Returns:
            Dictionary containing model information.
        """
        pass 