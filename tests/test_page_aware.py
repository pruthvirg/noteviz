"""
Simple test script for the page-aware PDF processor.
"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from noteviz.core.pdf import PDFConfig, PageAwarePDFProcessor, PageAwareChunk


async def test_page_aware_pdf():
    """Test the page-aware PDF processor."""
    # Create a mock PDF reader
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
        
        # Create a PDF processor
        config = PDFConfig(chunk_size=100, chunk_overlap=20)
        processor = PageAwarePDFProcessor(config)
        
        # Process the PDF
        pdf_path = Path("test.pdf")
        chunks = await processor.process_pdf(pdf_path)
        
        # Verify the chunks
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
            assert len(chunk.text) <= processor.config.chunk_size
            assert chunk.start_char < chunk.end_char
        
        # Test finding pages with a topic
        test_chunks = [
            PageAwareChunk("This is about AI.", 1, 0, 15),
            PageAwareChunk("AI is interesting.", 1, 16, 31),
            PageAwareChunk("Machine learning is part of AI.", 2, 0, 25),
            PageAwareChunk("Deep learning is also part of AI.", 2, 26, 50),
        ]
        
        # Test finding pages with "AI"
        ai_pages = processor.find_topic_pages(test_chunks, "AI")
        assert ai_pages == [1, 2]
        
        # Test finding pages with "machine learning"
        ml_pages = processor.find_topic_pages(test_chunks, "machine learning")
        assert ml_pages == [2]
        
        # Test finding pages with non-existent topic
        empty_pages = processor.find_topic_pages(test_chunks, "blockchain")
        assert empty_pages == []
        
        print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_page_aware_pdf()) 