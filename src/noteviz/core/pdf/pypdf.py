"""
pypdf implementation of the PDF processor.
"""
from pathlib import Path
from typing import List

from pypdf import PdfReader

from .base import PDFConfig, PDFProcessor


class PyPDFProcessor(PDFProcessor):
    """pypdf implementation of the PDF processor."""
    
    async def process_pdf(self, pdf_path: Path) -> List[str]:
        """Process a PDF file and return chunks of text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of text chunks.
        """
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extract text from all pages
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # Split text into chunks
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap
            
        return chunks
    
    async def extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary containing PDF metadata.
        """
        reader = PdfReader(pdf_path)
        metadata = reader.metadata
        
        return {
            "title": metadata.get("/Title", ""),
            "author": metadata.get("/Author", ""),
            "subject": metadata.get("/Subject", ""),
            "keywords": metadata.get("/Keywords", ""),
            "creator": metadata.get("/Creator", ""),
            "producer": metadata.get("/Producer", ""),
            "num_pages": len(reader.pages),
        } 