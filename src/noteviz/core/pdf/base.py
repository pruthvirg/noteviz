"""
Base interface for PDF processing services.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class PDFConfig(BaseModel):
    """Configuration for PDF processing."""
    chunk_size: int = 1000  # Number of characters per chunk
    chunk_overlap: int = 200  # Number of characters to overlap between chunks


class PDFProcessor(ABC):
    """Base class for PDF processing services."""
    
    def __init__(self, config: PDFConfig):
        self.config = config
    
    @abstractmethod
    async def process_pdf(self, pdf_path: Path) -> List[str]:
        """Process a PDF file and return chunks of text.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List of text chunks.
        """
        pass
    
    @abstractmethod
    async def extract_metadata(self, pdf_path: Path) -> dict:
        """Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary containing PDF metadata.
        """
        pass 