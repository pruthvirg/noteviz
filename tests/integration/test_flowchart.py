"""Integration tests for the flowchart generator."""
import os
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from dotenv import load_dotenv

from noteviz.core.flowchart import OpenAIFlowchartGenerator, MermaidRenderer
from noteviz.core.pdf import PDFConfig, PyPDFProcessor

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def test_pdf_path():
    """Create a test PDF file."""
    current_dir = Path(__file__).parent
    return current_dir / "data" / "sample.pdf"


@pytest.fixture
def pdf_processor():
    """Create a PDF processor."""
    config = PDFConfig(chunk_size=1000, chunk_overlap=200)
    return PyPDFProcessor(config)


@pytest.fixture
async def openai_client():
    """Create an OpenAI client."""
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set in environment")
    return AsyncOpenAI()


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.asyncio
async def test_flowchart_generation_from_pdf(test_pdf_path, pdf_processor, openai_client):
    """Test generating a flowchart from a PDF file."""
    # Process PDF
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    assert len(chunks) > 0
    text = "\n".join(chunks)

    # Generate flowchart
    generator = OpenAIFlowchartGenerator(client=openai_client)
    flowchart = await generator.generate_flowchart(text)

    # Verify flowchart structure
    assert flowchart.nodes, "Flowchart should have nodes"
    assert flowchart.edges, "Flowchart should have edges"
    assert all(0 <= node.confidence <= 1 for node in flowchart.nodes), "Node confidence scores should be between 0 and 1"
    assert all(node.description for node in flowchart.nodes), "Nodes should have descriptions"

    # Verify edge connections
    node_ids = {node.id for node in flowchart.nodes}
    for edge in flowchart.edges:
        assert edge.source in node_ids, f"Edge source {edge.source} not found in nodes"
        assert edge.target in node_ids, f"Edge target {edge.target} not found in nodes"


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.asyncio
async def test_flowchart_with_specific_topic(test_pdf_path, pdf_processor, openai_client):
    """Test generating a flowchart with a specific topic."""
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    text = "\n".join(chunks)

    generator = OpenAIFlowchartGenerator(client=openai_client)
    topic = "Neural Network Components"
    keywords = ["neurons", "weights", "activation functions"]
    flowchart = await generator.generate_flowchart(text, topic=topic, keywords=keywords)

    assert flowchart.title == f"Flowchart: {topic}"
    assert any(any(keyword.lower() in str(node).lower() for node in flowchart.nodes) 
              for keyword in keywords), "Keywords should be reflected in nodes"


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.asyncio
async def test_flowchart_to_mermaid(test_pdf_path, pdf_processor, openai_client):
    """Test converting a generated flowchart to Mermaid format."""
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    text = "\n".join(chunks)

    # Generate flowchart
    generator = OpenAIFlowchartGenerator(client=openai_client)
    flowchart = await generator.generate_flowchart(text)

    # Convert to Mermaid
    renderer = MermaidRenderer()
    mermaid = renderer.render(flowchart)

    # Verify Mermaid syntax
    assert mermaid.startswith("flowchart TD"), "Should start with flowchart declaration"
    assert all(node.id in mermaid for node in flowchart.nodes), "All nodes should be in Mermaid output"
    for edge in flowchart.edges:
        assert f"{edge.source} -->" in mermaid, "Edge connections should be in Mermaid output"


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.asyncio
async def test_large_document_flowchart(test_pdf_path, pdf_processor, openai_client):
    """Test generating a flowchart from a large document."""
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    # Duplicate chunks to simulate larger document
    chunks = chunks * 3
    text = "\n".join(chunks)

    generator = OpenAIFlowchartGenerator(client=openai_client)
    flowchart = await generator.generate_flowchart(text)

    # Verify reasonable output size
    assert 5 <= len(flowchart.nodes) <= 10, "Should have reasonable number of nodes"
    assert len(flowchart.edges) >= len(flowchart.nodes) - 1, "Should have sufficient connections"


@pytest.mark.expensive
@pytest.mark.skip(reason="Skipping expensive OpenAI API test")
@pytest.mark.asyncio
async def test_multiple_flowcharts_same_document(test_pdf_path, pdf_processor, openai_client):
    """Test generating multiple flowcharts from the same document with different topics."""
    chunks = await pdf_processor.process_pdf(test_pdf_path)
    text = "\n".join(chunks)

    generator = OpenAIFlowchartGenerator(client=openai_client)
    
    # Generate flowcharts for different topics
    topics = ["Topic 1", "Topic 2"]
    flowcharts = []
    
    for topic in topics:
        flowchart = await generator.generate_flowchart(text, topic=topic)
        flowcharts.append(flowchart)

    # Verify different flowcharts
    assert flowcharts[0].title != flowcharts[1].title, "Different topics should yield different flowcharts"
    assert set(node.id for node in flowcharts[0].nodes) != set(node.id for node in flowcharts[1].nodes), \
        "Different topics should have different nodes" 