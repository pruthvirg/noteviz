"""
OpenAI implementation of the flowchart generator.
"""
import json
from typing import List, Optional

from openai import AsyncOpenAI

from .base import FlowchartGenerator, Flowchart, Node, Edge
from ..llm.openai import OpenAITopicExtractor
from ..llm.config import TopicExtractorConfig


class OpenAIFlowchartGenerator(FlowchartGenerator):
    """OpenAI-based flowchart generator implementation."""
    
    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """Initialize the flowchart generator.
        
        Args:
            client: Optional OpenAI client. If not provided, a new one will be created.
        """
        self.client = client or AsyncOpenAI()
        # Pass the same client to the topic extractor
        self.topic_extractor = OpenAITopicExtractor(
            TopicExtractorConfig(
                model_name="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=2000,
                num_topics=5
            ),
            client=self.client
        )
    
    async def generate_flowchart(
        self,
        text: str,
        topic: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> Flowchart:
        """Generate a flowchart from text.
        
        Args:
            text: The text to analyze
            topic: Optional main topic to focus on. If not provided, will be extracted.
            keywords: Optional list of relevant keywords. If not provided, will be extracted.
            
        Returns:
            A Flowchart object containing nodes and edges
        """
        # Extract topics and keywords if not provided
        if topic is None or keywords is None:
            topics = await self.topic_extractor.extract_topics([text])
            if topic is None:
                topic = topics[0].name
            if keywords is None:
                keywords = topics[0].keywords

        prompt = f"""Given the text below, create a flowchart about {topic}.
        Focus on these keywords: {', '.join(keywords)}

        Text: {text}

        Create a flowchart that:
        1. Identifies key concepts as nodes (5-10 nodes)
        2. Shows relationships between concepts as edges
        3. Assigns confidence scores (0.0-1.0) to each concept
        4. Provides brief descriptions for nodes and edges

        Output the flowchart as a JSON structure:
        {{
            "nodes": [
                {{
                    "id": "unique_id",
                    "label": "concept name",
                    "description": "brief explanation",
                    "confidence": 0.95
                }}
            ],
            "edges": [
                {{
                    "source": "node_id",
                    "target": "node_id",
                    "label": "relationship type",
                    "description": "explanation of relationship"
                }}
            ]
        }}"""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a flowchart generator that creates structured visualizations of concepts and their relationships."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        data = json.loads(response.choices[0].message.content)

        # Convert JSON to Flowchart object
        nodes = [
            Node(
                id=node["id"],
                label=node["label"],
                description=node.get("description", ""),
                confidence=node["confidence"]
            )
            for node in data["nodes"]
        ]

        edges = [
            Edge(
                source=edge["source"],
                target=edge["target"],
                label=edge["label"],
                description=edge.get("description", "")
            )
            for edge in data["edges"]
        ]

        return Flowchart(
            nodes=nodes,
            edges=edges,
            title=f"Flowchart: {topic}",
            description=f"Flowchart generated from text about {topic}"
        ) 