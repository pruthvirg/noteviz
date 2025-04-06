# NoteViz

A tool for analyzing and visualizing books and their concepts using Large Language Models (LLMs). NoteViz helps you understand complex documents by extracting key topics, generating summaries, and creating visual representations of the content.

## Features

- PDF text extraction and processing
- Intelligent text summarization using OpenAI's GPT models
- Topic extraction and analysis with confidence scores
- Key concept identification
- Semantic search capabilities
- Command-line interface
- (Coming Soon) Flowchart generation for concept visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/noteviz.git
   cd noteviz
   ```

2. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   ```

4. Install the package:
   ```bash
   uv pip install -e .  # For basic installation
   uv pip install -e ".[dev]"  # For development with test dependencies
   ```

5. Set up your OpenAI API key:
   ```bash
   # On Unix/macOS
   export OPENAI_API_KEY=your_api_key_here
   
   # On Windows (PowerShell)
   $env:OPENAI_API_KEY = "your_api_key_here"
   ```

## Usage

Process a PDF file:
```bash
noteviz process path/to/your.pdf
```

This will:
1. Extract text from the PDF
2. Generate a comprehensive summary
3. Extract key topics with descriptions and confidence scores
4. Identify important concepts
5. (Coming Soon) Generate a flowchart visualization

## Development

### Running Tests
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/  # Unit tests
pytest tests/integration/  # Integration tests
```

### Project Structure
```
noteviz/
├── src/
│   └── noteviz/
│       ├── core/           # Core functionality
│       │   ├── pdf/        # PDF processing
│       │   ├── llm/        # LLM integration
│       │   ├── embedding/  # Text embedding
│       │   └── retrieval/  # Semantic search
│       ├── cli.py          # Command-line interface
│       └── __init__.py
├── tests/                  # Test files
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── pyproject.toml         # Project configuration
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
