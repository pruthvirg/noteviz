
def process_pdf_sync(self, pdf_path):
    """Process a PDF file synchronously and return chunks."""
    reader = PdfReader(pdf_path)
    chunks = []
    
    # Process each page separately to maintain page boundaries
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if not page_text.strip():
            continue
        
        # Split page text into chunks with a maximum size
        chunk_size = self.config.chunk_size
        overlap = min(self.config.chunk_overlap, chunk_size // 2)
        
        start = 0
        while start < len(page_text):
            # Calculate end position
            end = min(start + chunk_size, len(page_text))
            
            # Find a good breaking point
            if end < len(page_text):
                # Try to break at paragraph
                para_break = page_text.rfind('\n\n', start, end)
                if para_break != -1 and para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Try to break at sentence
                    sent_break = page_text.rfind('. ', start, end)
                    if sent_break != -1 and sent_break > start + chunk_size // 2:
                        end = sent_break + 1
            
            # Create chunk
            chunk_text = page_text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunk = PageAwareChunk(
                    text=chunk_text,
                    page_number=page_num,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            # Move start position, ensuring we make progress
            new_start = end - overlap
            if new_start <= start:  # Prevent infinite loop
                new_start = end
            start = new_start
            
            if start >= len(page_text):
                break
    
    return chunks 