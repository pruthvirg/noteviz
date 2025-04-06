"""Tests for Mermaid renderer."""
from noteviz.core.flowchart.base import Node, Edge, Flowchart
from noteviz.core.flowchart.mermaid import MermaidRenderer

def test_mermaid_renderer_init():
    """Test MermaidRenderer initialization."""
    # Test with default parameters
    renderer = MermaidRenderer()
    assert renderer.direction == "TD"
    assert renderer.theme == "default"
    assert renderer.node_style == {}
    assert renderer.edge_style == {}
    
    # Test with custom parameters
    renderer = MermaidRenderer(
        direction="LR",
        theme="dark",
        node_style={"fill": "#fff"},
        edge_style={"stroke": "#000"}
    )
    assert renderer.direction == "LR"
    assert renderer.theme == "dark"
    assert renderer.node_style == {"fill": "#fff"}
    assert renderer.edge_style == {"stroke": "#000"}

def test_mermaid_render_simple():
    """Test rendering a simple flowchart."""
    renderer = MermaidRenderer()
    flowchart = Flowchart(
        nodes=[
            Node(id="n1", label="Node 1"),
            Node(id="n2", label="Node 2")
        ],
        edges=[
            Edge(source="n1", target="n2", label="connects to")
        ],
        title="Simple Chart"
    )
    
    mermaid = renderer.render(flowchart)
    
    # Check basic structure
    assert "flowchart TD" in mermaid
    assert "%% Simple Chart" in mermaid
    assert 'n1["Node 1"]' in mermaid
    assert 'n2["Node 2"]' in mermaid
    assert 'n1 -->|"connects to"| n2' in mermaid

def test_mermaid_render_with_descriptions():
    """Test rendering a flowchart with descriptions."""
    renderer = MermaidRenderer()
    flowchart = Flowchart(
        nodes=[
            Node(id="n1", label="Node 1", description="First node"),
            Node(id="n2", label="Node 2", description="Second node")
        ],
        edges=[
            Edge(
                source="n1",
                target="n2",
                label="connects to",
                description="Connection description"
            )
        ],
        title="Detailed Chart",
        description="A test chart"
    )
    
    mermaid = renderer.render(flowchart)
    
    # Check descriptions
    assert "%% A test chart" in mermaid
    assert 'click n1 tooltip "First node"' in mermaid
    assert 'click n2 tooltip "Second node"' in mermaid
    assert 'click n1_n2 tooltip "Connection description"' in mermaid

def test_mermaid_render_with_confidence():
    """Test rendering nodes with different confidence levels."""
    renderer = MermaidRenderer()
    flowchart = Flowchart(
        nodes=[
            Node(id="n1", label="High", confidence=0.9),
            Node(id="n2", label="Medium", confidence=0.7),
            Node(id="n3", label="Low", confidence=0.5)
        ],
        edges=[],
        title="Confidence Test"
    )
    
    mermaid = renderer.render(flowchart)
    
    # Check confidence styles
    assert ":::highConfidence" in mermaid
    assert ":::mediumConfidence" in mermaid
    assert ":::lowConfidence" in mermaid
    assert "fill:#90EE90" in mermaid  # High confidence color
    assert "fill:#FFE4B5" in mermaid  # Medium confidence color
    assert "fill:#FFB6C1" in mermaid  # Low confidence color

def test_mermaid_render_with_custom_styles():
    """Test rendering with custom styles."""
    renderer = MermaidRenderer(
        node_style={"fill": "#ffffff", "stroke": "#000000"},
        edge_style={"stroke": "#666666"}
    )
    flowchart = Flowchart(
        nodes=[Node(id="n1", label="Test")],
        edges=[],
        title="Style Test"
    )
    
    mermaid = renderer.render(flowchart)
    
    # Check custom styles
    assert "fill:#ffffff" in mermaid
    assert "stroke:#000000" in mermaid 