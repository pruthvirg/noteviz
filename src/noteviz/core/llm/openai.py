"""
OpenAI implementation of the LLM service.
"""
import json
import random
from typing import List, Optional

from openai import AsyncOpenAI

from .base import LLMConfig, LLMService, Topic, Summarizer, TopicExtractor
from .config import SummarizerConfig, TopicExtractorConfig


class OpenAILLMService(LLMService):
    """OpenAI-based LLM service implementation."""
    
    def __init__(
        self,
        summarizer_config: SummarizerConfig,
        topic_extractor_config: TopicExtractorConfig,
        client: Optional[AsyncOpenAI] = None
    ):
        super().__init__(summarizer_config, topic_extractor_config)
        self.client = client or AsyncOpenAI()
    
    async def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        """Generate a summary of the text.
        
        Args:
            text: Text to summarize.
            max_length: Maximum length of the summary.
            
        Returns:
            Generated summary.
        """
        if not text:
            raise ValueError("No text provided for summarization")
            
        prompt = "Please summarize the following text:\n\n"
        if max_length:
            prompt += f"Keep the summary under {max_length} words.\n\n"
        prompt += text
        
        response = await self.client.chat.completions.create(
            model=self.summarizer_config.model_name,
            temperature=self.summarizer_config.temperature,
            max_tokens=self.summarizer_config.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    async def identify_key_concepts(self, text: str, num_concepts: int = 5) -> List[str]:
        """Identify key concepts in the text.
        
        Args:
            text: Text to analyze.
            num_concepts: Number of concepts to identify.
            
        Returns:
            List of key concepts.
        """
        if not text:
            raise ValueError("No text provided for concept identification")
            
        prompt = f"""Please identify the {num_concepts} most important concepts from the following text.
        For each concept, provide a brief explanation.
        Format the response as a numbered list.
        
        Text to analyze:
        {text}
        """
        
        response = await self.client.chat.completions.create(
            model=self.topic_extractor_config.model_name,
            temperature=self.topic_extractor_config.temperature,
            max_tokens=self.topic_extractor_config.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies key concepts in text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Split the response into lines and clean up
        concepts = response.choices[0].message.content.split("\n")
        return [concept.strip().split(". ", 1)[1] if ". " in concept else concept.strip() 
                for concept in concepts if concept.strip()]
    
    async def extract_topics(self, text: str, num_topics: int = 5) -> List[Topic]:
        """Extract main topics from text.
        
        Args:
            text: Text to analyze.
            num_topics: Number of topics to extract.
            
        Returns:
            List of extracted topics.
        """
        if not text:
            raise ValueError("No text provided for topic extraction")
            
        prompt = f"""Analyze the following text and extract EXACTLY {num_topics} topics, no more and no less.
        For each topic, provide:
        - name: A short, descriptive name
        - description: A brief explanation
        - confidence: A score between 0 and 1
        - keywords: A list of relevant keywords
        
        Format the response as a JSON array.
        
        Text to analyze:
        {text}
        """
        
        response = await self.client.chat.completions.create(
            model=self.topic_extractor_config.model_name,
            temperature=self.topic_extractor_config.temperature,
            max_tokens=self.topic_extractor_config.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts topics from text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        # Remove markdown code block formatting if present
        if "```json" in content:
            content = content.split("```json")[1]
        if "```" in content:
            content = content.split("```")[0]
        
        # Strip whitespace and parse JSON
        content = content.strip()
        topics_data = json.loads(content)
        
        return [
            Topic(
                name=topic["name"],
                description=topic["description"],
                confidence=topic["confidence"],
                keywords=topic["keywords"]
            )
            for topic in topics_data
        ]


class OpenAISummarizer(Summarizer):
    """OpenAI implementation of text summarization."""
    
    def __init__(self, config: SummarizerConfig, client: Optional[AsyncOpenAI] = None):
        super().__init__(config)
        self.client = client or AsyncOpenAI()
    
    async def summarize(self, text: str) -> str:
        """Generate a summary of the text.
        
        Args:
            text: Text to summarize.
            
        Returns:
            Generated summary.
        """
        prompt = "Please summarize the following text:\n\n"
        if self.config.max_summary_length:
            prompt += f"Keep the summary under {self.config.max_summary_length} words.\n\n"
        prompt += text
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content


class OpenAITopicExtractor(TopicExtractor):
    """OpenAI implementation of topic extraction."""
    
    def __init__(self, config: TopicExtractorConfig, client: Optional[AsyncOpenAI] = None):
        super().__init__(config)
        self.client = client or AsyncOpenAI()
    
    async def extract_topics(self, chunks: List[str]) -> List[Topic]:
        """Extract topics from text chunks.
        
        Args:
            chunks: List of text chunks to analyze.
            
        Returns:
            List of extracted topics.
            
        Raises:
            ValueError: If no text chunks are provided.
            json.JSONDecodeError: If the API response cannot be parsed.
        """
        if not chunks:
            raise ValueError("No text chunks provided")
        
        # Select chunks up to max_context_chunks
        selected_chunks = chunks[:self.config.max_context_chunks]
        combined_text = "\n\n".join(selected_chunks)
        
        prompt = f"""
        Analyze the following text and extract EXACTLY {self.config.num_topics} topics, no more and no less.
        For each topic, provide:
        - name: A short, descriptive name
        - description: A brief explanation
        - confidence: A score between 0 and 1
        - keywords: A list of relevant keywords
        
        Format the response as a JSON array.
        
        Text to analyze:
        {combined_text}
        """
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        content = response.choices[0].message.content
        
        # Remove markdown code block formatting if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        
        # Strip whitespace and parse JSON
        content = content.strip()
        topics_data = json.loads(content)
        
        return [
            Topic(
                name=topic["name"],
                description=topic["description"],
                confidence=topic["confidence"],
                keywords=topic["keywords"]
            )
            for topic in topics_data
        ] 