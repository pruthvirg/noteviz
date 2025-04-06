"""
Unit tests for the PDF processor.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from noteviz.core.pdf import PDFConfig
from noteviz.core.pdf.pypdf import PyPDFProcessor


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
    with patch('noteviz.core.pdf.pypdf.PdfReader') as mock:
        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "This is a test page content."
        reader.pages = [page, page]  # Two pages
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
    return PyPDFProcessor(pdf_config)


@pytest.mark.asyncio
async def test_process_pdf(pdf_processor, mock_pdf_reader):
    """Test PDF processing."""
    pdf_path = Path("test.pdf")
    chunks = await pdf_processor.process_pdf(pdf_path)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) <= pdf_processor.config.chunk_size for chunk in chunks)
    
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