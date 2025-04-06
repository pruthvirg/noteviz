"""
Command-line interface for NoteViz.
"""
import asyncio
import sys
from pathlib import Path

from noteviz.core.pdf import PDFConfig, PyPDFProcessor
from noteviz.core.embedding import EmbeddingConfig, OpenAIEmbeddingService
from noteviz.core.llm import (
    SummarizerConfig,
    TopicExtractorConfig,
    OpenAILLMService,
)
from noteviz.core.retrieval import RetrievalConfig, CosineRetrieval


async def process_pdf(pdf_path: str) -> dict:
    """Process a PDF file and generate analysis.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Dictionary containing analysis results.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} does not exist")
        sys.exit(1)
        
    # Initialize services
    pdf_config = PDFConfig(chunk_size=1000, chunk_overlap=200)
    pdf_processor = PyPDFProcessor(pdf_config)
    
    embedding_config = EmbeddingConfig(
        model_name="text-embedding-3-small",
        device="cpu",
        batch_size=32
    )
    embedding_service = OpenAIEmbeddingService(embedding_config)
    
    summarizer_config = SummarizerConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    topic_config = TopicExtractorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        num_topics=5
    )
    llm_service = OpenAILLMService(summarizer_config, topic_config)
    
    retrieval_config = RetrievalConfig(
        similarity_threshold=0.7,
        max_results=5
    )
    retrieval_service = CosineRetrieval(retrieval_config)
    
    # Process PDF
    print(f"Processing PDF: {pdf_path}")
    chunks = await pdf_processor.process_pdf(pdf_path)
    print(f"Extracted {len(chunks)} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = await embedding_service.generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Extract topics
    print("\nExtracting topics...")
    text = "\n".join(chunks)
    topics = await llm_service.extract_topics(text)
    print("\nTopics:")
    for topic in topics:
        print(f"- {topic.name}: {topic.description}")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = await llm_service.generate_summary(text)
    print(f"\nSummary:\n{summary}")
    
    # Identify key concepts
    print("\nIdentifying key concepts...")
    key_concepts = await llm_service.identify_key_concepts(text)
    print("\nKey Concepts:")
    for concept in key_concepts:
        print(f"- {concept}")
        
    return {
        "topics": topics,
        "summary": summary,
        "key_concepts": key_concepts
    }


def main(args=None):
    """Main entry point for the CLI."""
    if args is None:
        args = sys.argv[1:]
        
    if len(args) < 2:
        print("usage: noteviz [-h] {process} pdf_path")
        sys.exit(1)
        
    command = args[0]
    if command == "process":
        pdf_path = args[1]
        asyncio.run(process_pdf(pdf_path))
    else:
        print("usage: noteviz [-h] {process} pdf_path")
        print("noteviz: error: argument command: invalid choice: '{}' (choose from process)".format(command))
        sys.exit(1)


if __name__ == "__main__":
    main() 