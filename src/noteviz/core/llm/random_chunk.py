"""
Random chunk topic extractor implementation.
"""
import json
import random
from typing import List, Optional

from openai import AsyncOpenAI

from .base import TopicExtractor, Topic
from .config import TopicExtractorConfig


class RandomChunkTopicExtractor(TopicExtractor):
    """OpenAI implementation of topic extraction using random chunks."""
    
    def __init__(self, config: TopicExtractorConfig, client: Optional[AsyncOpenAI] = None):
        super().__init__(config)
        self.client = client or AsyncOpenAI()
    
    async def extract_topics(self, chunks: List[str]) -> List[Topic]:
        """Extract topics from randomly selected text chunks.
        
        Args:
            chunks: List of text chunks.
            
        Returns:
            List of extracted topics.
            
        Raises:
            ValueError: If no text chunks are provided.
            json.JSONDecodeError: If the API response cannot be parsed.
        """
        if not chunks:
            raise ValueError("No text chunks provided")
        
        # Randomly select chunks if we have more than 3
        if len(chunks) > 3:
            selected_chunks = random.sample(chunks, 3)
        else:
            selected_chunks = chunks
        
        # Combine text from selected chunks
        combined_text = "\n\n".join(selected_chunks)
        
        prompt = f"""Extract {self.config.num_topics} main topics from the following text chunks.
For each topic, provide:
1. A concise name
2. A brief description
3. A confidence score (0-1)
4. 2-3 relevant keywords

Text chunks:
{combined_text}

Return the topics in the following JSON format:
[
    {{
        "name": "Topic name",
        "description": "Topic description",
        "confidence": 0.95,
        "keywords": ["keyword1", "keyword2"]
    }}
]"""
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
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