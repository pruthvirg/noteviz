# NoteViz

A Streamlit application that generates flowcharts from PDF documents using LLMs and vector embeddings.

## Features

- PDF document processing and chunking
- Topic extraction using LLMs
- Vector-based and TF-IDF document retrieval
- Mermaid.js flowchart generation
- Interactive Streamlit UI

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd noteviz
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Set up configuration files:
```bash
# Copy application configuration
cp .env.example .env
# Edit .env with your application settings

# Copy secrets template
cp .secrets.example .secrets
# Edit .secrets with your API keys and sensitive information
```

5. Run the application:
```bash
streamlit run src/app.py
```

## Development

- Python 3.8+
- UV for dependency management
- Streamlit for UI
- OpenAI API for LLM
- SentenceTransformers for embeddings
- ChromaDB for vector storage

## Configuration

The application uses two configuration files:
- `.env`: Application settings and feature flags
- `.secrets`: API keys and sensitive information

## Testing

```bash
pytest
```

## License

MIT
