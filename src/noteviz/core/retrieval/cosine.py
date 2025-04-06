"""Cosine similarity-based retrieval service."""
from typing import List, Tuple
import numpy as np

from .base import RetrievalConfig, RetrievalService


class CosineRetrieval(RetrievalService):
    """Cosine similarity-based retrieval service."""
    
    def __init__(self, config: RetrievalConfig):
        super().__init__(config)
        self.texts = []
        self.embeddings = []
    
    def index(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Index the texts and their embeddings.
        
        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        if not texts:
            raise ValueError("No texts provided for indexing")
            
        self.texts = texts
        self.embeddings = embeddings
    
    def find_relevant_chunks(self, query_embedding: List[float]) -> List[Tuple[str, float]]:
        """Find relevant text chunks based on cosine similarity.
        
        Args:
            query_embedding: Embedding vector of the query.
            
        Returns:
            List of tuples containing (text, similarity_score).
        """
        if not self.texts or not self.embeddings:
            raise ValueError("No indexed texts available")
            
        # Convert embeddings to numpy arrays for efficient computation
        query_array = np.array(query_embedding)
        embeddings_array = np.array(self.embeddings)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_array, query_array) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array)
        )
        
        # Sort by similarity and filter by threshold
        indices = np.argsort(similarities)[::-1]
        results = []
        for idx in indices:
            similarity = similarities[idx]
            if similarity < self.config.similarity_threshold:
                break
            if len(results) >= self.config.max_results:
                break
            results.append((self.texts[idx], float(similarity)))
        
        return results 