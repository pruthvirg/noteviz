"""
Unit tests for the page-aware PDF processor.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from noteviz.core.pdf import PDFConfig, PageAwarePDFProcessor, PageAwareChunk

# Mark all tests in this module as asyncio tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def pdf_config():
    """Create a test PDF configuration."""
    return PDFConfig(
        chunk_size=100,
        chunk_overlap=20
    )


@pytest.fixture
def mock_pdf_reader():
    """Create a mock PDF reader."""
    with patch('noteviz.core.pdf.page_aware_pdf.PdfReader') as mock:
        reader = MagicMock()
        page1 = MagicMock()
        page1.extract_text.return_value = "This is page 1. It has multiple sentences. This is the last sentence."
        page2 = MagicMock()
        page2.extract_text.return_value = "This is page 2. It also has multiple sentences. This is the last sentence."
        reader.pages = [page1, page2]  # Two pages
        reader.metadata = {
            "/Title": "Test PDF",
            "/Author": "Test Author",
            "/Subject": "Test Subject",
            "/Keywords": "test, pdf",
            "/Creator": "Test Creator",
            "/Producer": "Test Producer"
        }
        mock.return_value = reader
        yield mock


@pytest.fixture
def pdf_processor(pdf_config):
    """Create a test PDF processor."""
    return PageAwarePDFProcessor(pdf_config)


@pytest.mark.asyncio
async def test_process_pdf(pdf_processor, mock_pdf_reader):
    """Test PDF processing with page awareness."""
    pdf_path = Path("test.pdf")
    chunks = await pdf_processor._process_pdf_with_pages(pdf_path)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, PageAwareChunk) for chunk in chunks)
    
    # Verify page numbers are correct
    page1_chunks = [chunk for chunk in chunks if chunk.page_number == 1]
    page2_chunks = [chunk for chunk in chunks if chunk.page_number == 2]
    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    
    # Verify chunk sizes
    for chunk in chunks:
        assert len(chunk.text) <= pdf_processor.config.chunk_size
        assert chunk.start_char < chunk.end_char
    
    # Verify reader was called correctly
    mock_pdf_reader.assert_called_once_with(pdf_path)


@pytest.mark.asyncio
async def test_extract_metadata(pdf_processor, mock_pdf_reader):
    """Test metadata extraction."""
    pdf_path = Path("test.pdf")
    metadata = await pdf_processor.extract_metadata(pdf_path)
    
    assert isinstance(metadata, dict)
    assert metadata["title"] == "Test PDF"
    assert metadata["author"] == "Test Author"
    assert metadata["subject"] == "Test Subject"
    assert metadata["keywords"] == "test, pdf"
    assert metadata["creator"] == "Test Creator"
    assert metadata["producer"] == "Test Producer"
    assert metadata["num_pages"] == 2
    
    # Verify reader was called correctly
    mock_pdf_reader.assert_called_once_with(pdf_path)


def test_find_topic_pages(pdf_processor):
    """Test finding pages containing a topic."""
    chunks = [
        PageAwareChunk("This is about AI.", 1, 0, 15),
        PageAwareChunk("AI is interesting.", 1, 16, 31),
        PageAwareChunk("Machine learning is part of AI.", 2, 0, 25),
        PageAwareChunk("Deep learning is also part of AI.", 2, 26, 50),
    ]
    
    # Test finding pages with "AI"
    ai_pages = pdf_processor.find_topic_pages(chunks, "AI")
    assert ai_pages == [1, 2]
    
    # Test finding pages with "machine learning"
    ml_pages = pdf_processor.find_topic_pages(chunks, "machine learning")
    assert ml_pages == [2]
    
    # Test finding pages with non-existent topic
    empty_pages = pdf_processor.find_topic_pages(chunks, "blockchain")
    assert empty_pages == []


@pytest.mark.asyncio
async def test_process_pdf_returns_strings(pdf_processor, mock_pdf_reader):
    """Test that process_pdf returns a list of strings."""
    pdf_path = Path("test.pdf")
    chunks = await pdf_processor.process_pdf(pdf_path)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    
    # Verify reader was called correctly
    mock_pdf_reader.assert_called_once_with(pdf_path) 