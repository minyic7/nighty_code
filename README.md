# Nighty Code

A modular Python framework for analyzing and understanding code repositories using intelligent parser selection - combining fast tree-sitter parsers with flexible LLM fallback for universal file support.

## 🎯 Key Features

- **Intelligent Parser Selection**: Automatically chooses between tree-sitter (fast, free) and LLM (flexible, universal)
- **Universal File Support**: Handles any text-based file format through dynamic entity models
- **Identity Cards**: Rich metadata cards for every file including dependencies and relationships
- **Cost Optimization**: Prefers free tree-sitter parsing, falls back to LLM only when needed
- **Structured Extraction**: Extract custom information using user-defined Pydantic schemas
- **Comprehensive Analysis**: Entities, relationships, dependencies, and repository-wide graphs

## 📊 Project Status

### ✅ Completed Components

- **Core Infrastructure**: Scanner, classifier, file type detection
- **Tree-sitter Parsers**: Scala (fully implemented), Python/Java/JS (placeholders)
- **LLM Integration**: Anthropic Claude support with token management
- **Dynamic Models**: Flexible entity types that accommodate any file format
- **Identity Cards**: Version 3.0 with upstream/downstream dependencies
- **Extraction Pipeline**: Intelligent parser selection with fallback strategy
- **Storage System**: JSON-based artifact storage with organized structure

### 🚧 In Progress

- Additional tree-sitter parsers
- Neo4j graph database integration
- MCP (Model Context Protocol) server
- Web UI for visualization

## 🏗️ Architecture

```
nighty_code/
├── src/nighty_code/
│   ├── core/           # Scanner, classifier, base models
│   ├── parsers/        # Language parsers (tree-sitter + LLM)
│   │   ├── tree_sitter/  # Fast, deterministic parsers
│   │   ├── llm/          # Flexible LLM-based parser
│   │   └── model/        # Entity and relationship models
│   ├── identity/       # Identity card generation with LLM summaries
│   ├── extraction/     # Structured extraction with schema support
│   ├── storage/        # Artifact persistence
│   ├── llm/           # LLM client implementations
│   └── graph/         # Graph building (Neo4j ready)
├── tests/             # Comprehensive test suite
├── scripts/           # Utility scripts
├── docs/              # Documentation
└── examples/          # Usage examples
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nighty_code.git
cd nighty_code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

```bash
# Create .env file for API keys
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### Basic Usage

```python
from nighty_code.extraction import StructuredExtractor, ExtractionConfig
from pathlib import Path

# Configure extraction
config = ExtractionConfig(
    use_tree_sitter_when_available=True,  # Fast, free parsing
    use_llm_fallback=True,                # Universal coverage
    generate_identity_cards=True          # Rich metadata
)

# Extract from repository
extractor = StructuredExtractor(config)
response = extractor.extract_from_repository(
    repository_path=Path("path/to/repo"),
    max_files=None  # Process all files
)

print(f"Processed {response.files_processed} files")
print(f"Tree-sitter used: {response.parser_usage['tree_sitter']} files")
print(f"LLM used: {response.parser_usage['llm']} files")
```

## 🔧 Parser Strategy

The system uses an intelligent parser selection strategy:

1. **Is file supported by tree-sitter?** → Use tree-sitter (fast, accurate, free)
2. **Is file a known config/script format?** → Use LLM with specialized prompts
3. **Unknown file type?** → Use LLM as universal fallback

### Supported Tree-sitter Languages
- ✅ Scala (fully implemented)
- 🚧 Python, Java, JavaScript, TypeScript, Go (placeholders)

### LLM-Supported Formats
- ✅ Any text-based file format
- ✅ YAML, JSON, XML, TOML configurations
- ✅ SQL, Dockerfile, Shell scripts
- ✅ Markdown, documentation files

## 📈 Performance

Based on elt.aan.aan repository analysis:
- **18 files processed**: 100% success rate
- **4 Scala files**: Parsed with tree-sitter in <50ms each
- **14 other files**: Parsed with LLM in 5-10s each
- **Cost savings**: ~21% reduction in API calls through intelligent selection

## 🧩 Dynamic Entity Models

The system uses flexible entity types to handle any file format:

```python
# Instead of rigid enums:
class EntityType(Enum):
    CLASS = "class"
    FUNCTION = "function"
    # Limited to predefined types...

# We use dynamic models:
class DynamicEntity:
    entity_type: str  # Any string is valid
    # LLM can produce: "view", "stored_procedure", "workflow", etc.
```

## 📝 Identity Cards

Each file gets a comprehensive identity card with:
- File metadata and classification
- Purpose description
- Upstream dependencies (files it depends on)
- Downstream dependencies (files that depend on it)
- List of entities defined
- LLM-generated summary (optional)

## 🤝 Contributing

Contributions are welcome! Areas of focus:
1. Implementing more tree-sitter parsers
2. Improving LLM prompts for specific file types
3. Adding visualization components
4. Enhancing the Neo4j integration

## 📄 License

MIT License - see LICENSE file for details

## 🔮 Roadmap

### Phase 1: Core Infrastructure ✅
- Repository scanning
- File classification
- Basic entity extraction

### Phase 2: Parser Framework ✅
- Tree-sitter integration
- LLM fallback system
- Dynamic entity models

### Phase 3: Identity & Relationships ✅
- Identity card generation
- Dependency tracking
- LLM summaries

### Phase 4: Graph Database 🚧
- Neo4j integration
- Graph visualization
- Query interface

### Phase 5: API & Tools 🚧
- MCP server
- REST API
- Web UI

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is under active development. APIs may change between versions.