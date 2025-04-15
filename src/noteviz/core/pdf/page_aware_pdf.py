"""
Page-aware PDF processor implementation.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass

from pypdf import PdfReader

from .base import PDFConfig, PDFProcessor


@dataclass
class PageAwareChunk:
    """A chunk of text with its associated page number."""
    text: str
    page_number: int
    start_char: int
    end_char: int


class PageAwarePDFProcessor(PDFProcessor):
    """PDF processor that tracks page numbers for each chunk of text."""
    
    async def process_pdf(self, pdf_path: Path) -> List[str]:
        """Process a PDF file and return chunks of text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of text chunks.
        """
        # Get the page-aware chunks
        page_aware_chunks = await self._process_pdf_with_pages(pdf_path)
        
        # Extract just the text from each chunk to match the base class interface
        return [chunk.text for chunk in page_aware_chunks]
    
    async def _process_pdf_with_pages(self, pdf_path: Path) -> List[PageAwareChunk]:
        """Process a PDF file and return chunks of text with page numbers.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of PageAwareChunk objects containing text and page numbers.
        """
        reader = PdfReader(pdf_path)
        chunks = []
        
        # Process each page separately to maintain page boundaries
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if not page_text.strip():
                continue
                
            # Split page text into chunks
            start = 0
            while start < len(page_text):
                end = min(start + self.config.chunk_size, len(page_text))
                
                # Find a good breaking point (end of sentence or paragraph)
                if end < len(page_text):
                    # Try to break at paragraph
                    para_break = page_text.rfind('\n\n', start, end)
                    if para_break != -1 and para_break > start + self.config.chunk_size // 2:
                        end = para_break + 2
                    else:
                        # Try to break at sentence
                        sent_break = page_text.rfind('. ', start, end)
                        if sent_break != -1 and sent_break > start + self.config.chunk_size // 2:
                            end = sent_break + 1
                
                chunk = PageAwareChunk(
                    text=page_text[start:end].strip(),
                    page_number=page_num,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                
                # Move start position, considering overlap
                start = end - self.config.chunk_overlap
                if start >= len(page_text):
                    break
                    
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
    
    def find_topic_pages(self, chunks: List[PageAwareChunk], topic_text: str) -> List[int]:
        """Find pages that contain the given topic text.
        
        Args:
            chunks: List of PageAwareChunk objects.
            topic_text: Text to search for.
            
        Returns:
            List of page numbers where the topic appears.
        """
        topic_pages = set()
        topic_text = topic_text.lower()
        
        for chunk in chunks:
            if topic_text in chunk.text.lower():
                topic_pages.add(chunk.page_number)
                
        return sorted(list(topic_pages)) 