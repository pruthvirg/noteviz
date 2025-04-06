"""
Base interface for LLM services.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel

from .config import LLMConfig


class Topic:
    """Represents a topic extracted from text."""
    def __init__(self, name: str, description: str, confidence: float, keywords: List[str]):
        self.name = name
        self.description = description
        self.confidence = confidence
        self.keywords = keywords


class BaseLLMService(ABC):
    """Base class for LLM services."""
    def __init__(self, config: LLMConfig):
        self.config = config


class Summarizer(BaseLLMService):
    """Base class for text summarization."""
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config
    
    @abstractmethod
    async def summarize(self, text: str) -> str:
        """Generate a summary of the text.
        
        Args:
            text: Text to summarize.
            
        Returns:
            Generated summary.
        """
        pass


class TopicExtractor(BaseLLMService):
    """Base class for topic extraction."""
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.config = config
    
    @abstractmethod
    async def extract_topics(self, chunks: List[str]) -> List[Topic]:
        """Extract topics from text chunks.
        
        Args:
            chunks: List of text chunks to analyze.
            
        Returns:
            List of extracted topics.
        """
        pass


class LLMService(ABC):
    """Base class for LLM services."""
    
    def __init__(self, summarizer_config: LLMConfig, topic_extractor_config: LLMConfig):
        self.summarizer_config = summarizer_config
        self.topic_extractor_config = topic_extractor_config
    
    @abstractmethod
    async def extract_topics(self, text: str, num_topics: int = 5) -> List[Topic]:
        """Extract main topics from text.
        
        Args:
            text: Text to analyze.
            num_topics: Number of topics to extract.
            
        Returns:
            List of extracted topics.
        """
        pass
    
    @abstractmethod
    async def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        """Generate a summary of the text.
        
        Args:
            text: Text to summarize.
            max_length: Maximum length of the summary.
            
        Returns:
            Generated summary.
        """
        pass
    
    @abstractmethod
    async def identify_key_concepts(self, text: str, num_concepts: int = 5) -> List[str]:
        """Identify key concepts in the text.
        
        Args:
            text: Text to analyze.
            num_concepts: Number of concepts to identify.
            
        Returns:
            List of key concepts.
        """
        pass 