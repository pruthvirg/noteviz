"""Tests for flowchart base classes."""
import pytest
from typing import List
from noteviz.core.flowchart.base import Node, Edge, Flowchart, FlowchartGenerator

def test_node_creation():
    """Test creating a Node with various parameters."""
    # Test with required parameters
    node = Node(id="n1", label="Test Node")
    assert node.id == "n1"
    assert node.label == "Test Node"
    assert node.description is None
    assert node.confidence == 1.0
    
    # Test with all parameters
    node = Node(
        id="n2",
        label="Full Node",
        description="A test node",
        confidence=0.8
    )
    assert node.id == "n2"
    assert node.label == "Full Node"
    assert node.description == "A test node"
    assert node.confidence == 0.8

def test_edge_creation():
    """Test creating an Edge with various parameters."""
    # Test with required parameters
    edge = Edge(source="n1", target="n2", label="connects to")
    assert edge.source == "n1"
    assert edge.target == "n2"
    assert edge.label == "connects to"
    assert edge.description is None
    
    # Test with all parameters
    edge = Edge(
        source="n3",
        target="n4",
        label="leads to",
        description="A test edge"
    )
    assert edge.source == "n3"
    assert edge.target == "n4"
    assert edge.label == "leads to"
    assert edge.description == "A test edge"

def test_flowchart_creation():
    """Test creating a Flowchart with various parameters."""
    nodes = [
        Node(id="n1", label="Node 1"),
        Node(id="n2", label="Node 2")
    ]
    edges = [
        Edge(source="n1", target="n2", label="connects to")
    ]
    
    # Test with required parameters
    flowchart = Flowchart(nodes=nodes, edges=edges, title="Test Chart")
    assert flowchart.nodes == nodes
    assert flowchart.edges == edges
    assert flowchart.title == "Test Chart"
    assert flowchart.description is None
    
    # Test with all parameters
    flowchart = Flowchart(
        nodes=nodes,
        edges=edges,
        title="Full Chart",
        description="A test flowchart"
    )
    assert flowchart.nodes == nodes
    assert flowchart.edges == edges
    assert flowchart.title == "Full Chart"
    assert flowchart.description == "A test flowchart"

class TestFlowchartGenerator(FlowchartGenerator):
    """Test implementation of FlowchartGenerator."""
    async def generate_flowchart(
        self,
        text: str,
        topic: str,
        keywords: List[str]
    ) -> Flowchart:
        """Generate a test flowchart."""
        return Flowchart(
            nodes=[Node(id="n1", label="Test")],
            edges=[],
            title=topic
        )

@pytest.mark.asyncio
async def test_flowchart_generator():
    """Test the FlowchartGenerator interface."""
    generator = TestFlowchartGenerator()
    flowchart = await generator.generate_flowchart(
        text="test text",
        topic="Test Topic",
        keywords=["test"]
    )
    assert isinstance(flowchart, Flowchart)
    assert flowchart.title == "Test Topic"
    assert len(flowchart.nodes) == 1
    assert len(flowchart.edges) == 0 