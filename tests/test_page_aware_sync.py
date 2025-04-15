"""
Synchronous test script for the page-aware PDF processor.
"""
from noteviz.core.pdf.page_aware_pdf import PageAwarePDFProcessor, PageAwareChunk
from noteviz.core.pdf.base import PDFConfig


def test_find_topic_pages():
    """Test the find_topic_pages method of the PageAwarePDFProcessor."""
    # Create a PDF processor
    config = PDFConfig()
    processor = PageAwarePDFProcessor(config)
    
    # Create some test chunks with page numbers
    chunks = [
        PageAwareChunk("This is about AI and machine learning", page_number=1, start_char=0, end_char=35),
        PageAwareChunk("Deep learning is a subset of AI", page_number=2, start_char=0, end_char=35),
        PageAwareChunk("Neural networks are used in deep learning", page_number=3, start_char=0, end_char=40)
    ]
    
    # Test finding pages for a topic that exists
    pages = processor.find_topic_pages(chunks, "AI")
    assert pages == [1, 2], f"Expected pages [1, 2] for topic 'AI', got {pages}"
    
    # Test finding pages for a topic that appears in multiple chunks
    pages = processor.find_topic_pages(chunks, "machine learning")
    assert pages == [1], f"Expected pages [1] for topic 'machine learning', got {pages}"
    
    # Test finding pages for a topic that doesn't exist
    pages = processor.find_topic_pages(chunks, "blockchain")
    assert pages == [], f"Expected empty pages for non-existent topic, got {pages}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_find_topic_pages() 