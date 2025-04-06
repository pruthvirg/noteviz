"""Base classes for flowchart generation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Node:
    """A node in the flowchart representing a concept."""
    id: str
    label: str
    description: Optional[str] = None
    confidence: float = 1.0

@dataclass
class Edge:
    """An edge in the flowchart representing a relationship between concepts."""
    source: str
    target: str
    label: str
    description: Optional[str] = None

@dataclass
class Flowchart:
    """A complete flowchart with nodes and edges."""
    nodes: List[Node]
    edges: List[Edge]
    title: str
    description: Optional[str] = None

class FlowchartGenerator(ABC):
    """Abstract base class for flowchart generators."""
    
    @abstractmethod
    async def generate_flowchart(
        self,
        text: str,
        topic: str,
        keywords: List[str]
    ) -> Flowchart:
        """Generate a flowchart from text.
        
        Args:
            text: The text to analyze
            topic: The main topic to focus on
            keywords: List of relevant keywords
            
        Returns:
            A Flowchart object containing nodes and edges
        """
        pass 