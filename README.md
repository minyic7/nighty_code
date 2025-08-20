# Nighty Code

An intelligent code analysis framework that provides repository-aware copilot functionality using cached artifacts and LLM integration.

## Features

- **Repository Analysis**: Extracts code structure, dependencies, and relationships
- **Intelligent Q&A**: Answer questions about your codebase using LLM (Claude)
- **Artifact Caching**: Saves analysis results to avoid reprocessing
- **MCP Tools**: Selective context loading to prevent token explosion
- **Interactive Copilot**: Chat interface for repository exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nighty_code.git
cd nighty_code

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Quick Start

### 1. Generate Artifacts
```bash
python setup_artifacts.py
```

### 2. Test the Copilot
```bash
python test_copilot.py
```

### 3. Interactive Session
```bash
python test_copilot_interactive.py

# Or analyze your own repository
python test_copilot_interactive.py /path/to/your/repo
```

## Usage in Code

```python
from nighty_code.copilot import CopilotClient

# Initialize copilot for a repository
copilot = CopilotClient(
    repository_path="path/to/repo",
    llm_provider="anthropic",
    llm_model="claude-3-5-haiku-20241022"
)

# Ask questions
answer = copilot.ask("How do I use this repository?")
print(answer)

# Search for code elements
results = copilot.search_code(["export", "process"])

# Get file information
info = copilot.get_file_info("src/main.py")
```

## Architecture

```
nighty_code/
├── core/               # Core functionality (repository context, artifact management)
├── copilot/            # Main copilot interface with LLM integration
├── mcp/                # Model Context Protocol tools for artifact queries
├── llm/                # LLM client (Anthropic Claude)
├── extraction/         # Code extraction and analysis
├── parsers/            # Language-specific parsing (Scala, Python, etc.)
├── storage/            # Artifact storage and formats
└── identity/           # Identity card generation for files
```

## Key Components

- **CopilotClient**: Main interface for repository Q&A
- **RepositoryContext**: Manages artifact loading and caching
- **ArtifactTools**: MCP tools for selective context queries
- **LLMClient**: Unified interface for LLM providers

## Example Questions

The copilot can answer questions like:
- "How do I use this repository?"
- "What is the project structure?"
- "What are the main components?"
- "Show me all configuration files"
- "How does the export functionality work?"
- "What are the dependencies?"

## Environment Variables

```bash
ANTHROPIC_API_KEY=your_api_key      # Required for LLM features
LLM_PROVIDER=anthropic               # Default: anthropic
LLM_MODEL=claude-3-5-haiku-20241022 # Default model
```

## Development

```bash
# Run tests
pytest tests/

# Generate artifacts for sample repository
python setup_artifacts.py

# Test with sample repository
python test_copilot.py
```

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first.