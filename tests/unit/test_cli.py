"""Tests for the command-line interface."""
import pytest
import shutil
from unittest.mock import AsyncMock, patch
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from pypdf import PdfReader

from noteviz.cli import process_pdf, main
from noteviz.core.llm import Topic


# Configure test directories
TEST_DATA_DIR = Path("tests/data")
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Only mark async functions with asyncio
pytestmark = []


@pytest.fixture
def test_pdf_path():
    """Create a test PDF file with sample content.
    
    This fixture generates a PDF document with formatted text content
    that can be used to test the PDF processing capabilities of the CLI.
    The PDF includes:
    - A title in 24pt font
    - Multiple paragraphs of body text in 12pt font
    - Proper margins and spacing
    
    Returns:
        Path: Path to the generated test PDF file
    """
    permanent_pdf_path = TEST_DATA_DIR / "test.pdf"
    
    print(f"\nCreating test PDF at: {permanent_pdf_path}")
    
    # Create a PDF document
    doc = SimpleDocTemplate(
        str(permanent_pdf_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    )
    
    # Create content
    content = []
    content.append(Paragraph("Test PDF Document", title_style))
    content.append(Paragraph("This is a test document for CLI testing.", body_style))
    content.append(Paragraph("It contains multiple paragraphs of text that will be processed by our CLI.", body_style))
    content.append(Paragraph("The document includes various sections and formatting to test our PDF processing capabilities.", body_style))
    
    # Build the PDF
    doc.build(content)
    print(f"PDF created successfully at: {permanent_pdf_path}")
    print(f"PDF exists: {permanent_pdf_path.exists()}")
    print(f"PDF size: {permanent_pdf_path.stat().st_size} bytes")
    
    yield permanent_pdf_path
    
    # Note: We're keeping the PDF for inspection
    print(f"PDF remains at: {permanent_pdf_path}")


@pytest.fixture
def mock_services():
    """Mock the OpenAI services for testing."""
    with patch("noteviz.cli.OpenAIEmbeddingService") as mock_embedding, \
         patch("noteviz.cli.OpenAILLMService") as mock_llm, \
         patch("noteviz.cli.CosineRetrieval") as mock_retrieval:
        
        # Mock embedding service
        mock_embedding_instance = AsyncMock()
        mock_embedding_instance.generate_embeddings.return_value = [[0.1] * 1536] * 2
        mock_embedding.return_value = mock_embedding_instance
        
        # Mock LLM service
        mock_llm_instance = AsyncMock()
        mock_llm_instance.generate_summary.return_value = "This is a test summary."
        mock_llm_instance.identify_key_concepts.return_value = ["concept1", "concept2"]
        mock_llm_instance.extract_topics.return_value = [
            Topic(name="Test Topic", description="A test topic", confidence=0.9, keywords=["test"])
        ]
        mock_llm.return_value = mock_llm_instance
        
        # Mock retrieval service
        mock_retrieval_instance = AsyncMock()
        mock_retrieval_instance.find_relevant_chunks.return_value = []
        mock_retrieval.return_value = mock_retrieval_instance
        
        yield {
            "embedding": mock_embedding_instance,
            "llm": mock_llm_instance,
            "retrieval": mock_retrieval_instance
        }


@pytest.mark.asyncio
async def test_process_pdf(test_pdf_path, mock_services):
    """Test processing a PDF file."""
    result = await process_pdf(test_pdf_path)
    assert result is not None
    assert isinstance(result, dict)
    assert "topics" in result
    assert "summary" in result
    assert "key_concepts" in result
    assert len(result["topics"]) == 1


def test_main_invalid_command():
    """Test main function with invalid command."""
    with pytest.raises(SystemExit):
        main(["invalid", "test.pdf"])


def test_main_process_command(test_pdf_path, mock_services):
    """Test main function with process command."""
    main(["process", str(test_pdf_path)])


def test_pdf_file_is_valid(test_pdf_path):
    """Test that the generated PDF file is valid and contains expected content."""
    print(f"\nTesting PDF at: {test_pdf_path}")
    print(f"PDF exists: {test_pdf_path.exists()}")
    print(f"PDF size: {test_pdf_path.stat().st_size} bytes")
    
    # Read the PDF and verify its contents
    reader = PdfReader(test_pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    print(f"Number of pages: {len(reader.pages)}")
    print(f"Extracted text: {text[:100]}...")
    
    assert "Test PDF Document" in text
    assert "This is a test document for CLI testing" in text 