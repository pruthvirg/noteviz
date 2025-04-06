"""
OpenAI implementation of the embedding service.
"""
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from .base import EmbeddingConfig, EmbeddingService


class OpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OpenAI embedding service."""
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI-based embedding service implementation."""
    
    def __init__(self, config: EmbeddingConfig, client: Optional[AsyncOpenAI] = None):
        super().__init__(config)
        self.client = client or AsyncOpenAI()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's API.
        
        Args:
            texts: List of text strings to generate embeddings for.
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        embeddings = []
        for text in texts:
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    async def get_model_info(self) -> dict:
        """Get information about the OpenAI embedding model."""
        return {
            "provider": "OpenAI",
            "model": self.config.model_name,
            "dimensions": 1536  # OpenAI's embedding dimension
        } 