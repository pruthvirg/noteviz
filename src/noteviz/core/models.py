"""
Data models for the NoteViz application.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a PDF document."""
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Extracted text content from the PDF")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class TextChunk(BaseModel):
    """Represents a chunk of text from a document."""
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Chunk metadata")
    importance: float = Field(default=1.0, description="Importance score of the chunk")


class Topic(BaseModel):
    """Represents a topic extracted from documents."""
    id: str = Field(..., description="Unique identifier for the topic")
    title: str = Field(..., description="Title of the topic")
    description: str = Field(..., description="Description of the topic")
    related_topics: List[str] = Field(default_factory=list, description="IDs of related topics")


class FlowchartResult(BaseModel):
    """Represents a generated flowchart."""
    id: str = Field(..., description="Unique identifier for the flowchart")
    topic_id: str = Field(..., description="ID of the topic this flowchart represents")
    mermaid_code: str = Field(..., description="Mermaid.js code for the flowchart")
    detail_level: str = Field(..., description="Level of detail in the flowchart")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp") 