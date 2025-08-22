# Nighty Code

An intelligent code analysis and exploration framework powered by Large Language Models (LLMs) and Model Context Protocol (MCP).

## Overview

Nighty Code provides a sophisticated AI-powered copilot for understanding, navigating, and analyzing codebases. It combines LLM capabilities with intelligent tool execution through MCP to deliver context-aware assistance for software development tasks.

## Features

### 🤖 Intelligent Copilot (`nighty_code.copilot`)
- **AI-Powered Code Analysis**: Leverage LLMs to understand and navigate codebases
- **Multi-Persona Support**: Choose from different AI personas (default, architect, security, performance)
- **Context-Aware Responses**: Automatic project structure analysis and context building
- **Interactive Chat Sessions**: Engage in conversational code exploration with memory persistence
- **Session Management**: Save, resume, and track conversation sessions

### 🧠 Unified LLM Client (`nighty_code.llm`)
- **Multi-Provider Support**: Works with Anthropic Claude and OpenAI models
- **Token Management**: Automatic token counting, validation, and budget control
- **Smart Continuation**: Handles long outputs with automatic continuation
- **Cost Tracking**: Monitor API usage and costs
- **Structured Output**: Support for Pydantic models and structured responses

### 🔧 Model Context Protocol (`nighty_code.mcp`)
- **Tool Orchestration**: Intelligent tool selection and execution
- **File Operations**: Read files, list directories, search patterns
- **Smart Search**: Fuzzy file finding and code pattern matching
- **Intent Recognition**: Understand user queries and map to appropriate tools
- **Async Execution**: Efficient parallel tool execution

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nighty_code.git
cd nighty_code

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider Configuration
LLM_PROVIDER=anthropic  # or "openai"
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here  # If using OpenAI

# Model Selection
LLM_MODEL=claude-3-5-haiku-20241022  # or any supported model

# Optional Settings
LLM_TRACK_COSTS=true
LLM_LOG_TOKENS=true
```

## Usage

### Quick Start

```python
from nighty_code.copilot import CopilotClient

# Initialize copilot for your project
copilot = CopilotClient(
    folder_path="/path/to/your/project",
    persona_type="architect",  # or "default", "security", "performance"
)

# Ask questions about your codebase
response = copilot.ask("What does the main function do?")
print(response)

# Start an interactive chat session
copilot.chat()
```

### Using the LLM Client Directly

```python
from nighty_code.llm import LLMClient, LLMConfig, LLMProvider

# Configure the LLM client
config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    api_key="your_api_key",
    model="claude-3-5-haiku-20241022"
)

# Create client and get completions
client = LLMClient(config)
response = client.complete(
    prompt="Explain this code...",
    system_prompt="You are a helpful coding assistant.",
    max_tokens=2000
)
print(response.content)
```

### Using MCP Tools

```python
from nighty_code.mcp import MCPServer
import asyncio

# Initialize MCP server
server = MCPServer(project_root="/path/to/project")

# Execute tools
async def explore():
    request = {
        "method": "tool/list_directory",
        "params": {"directory_path": "."}
    }
    result = await server.handle_request(request)
    print(result)

asyncio.run(explore())
```

## Architecture

### Module Structure

```
nighty_code/
├── copilot/           # AI Copilot implementation
│   ├── client.py      # Main CopilotClient
│   ├── core/          # Core components
│   │   ├── intent.py      # Intent recognition
│   │   ├── orchestrator.py # Query orchestration
│   │   ├── tools.py       # Tool execution
│   │   └── validator.py   # Input validation
│   ├── memory/        # Memory management
│   │   ├── manager.py     # Memory manager
│   │   └── session.py     # Session management
│   └── models/        # Data models
│       └── persona.py     # AI personas
│
├── llm/               # LLM integration layer
│   ├── client.py      # Main LLM client
│   ├── config.py      # Configuration models
│   ├── structured_client.py # Structured output support
│   └── token_utils.py # Token counting utilities
│
└── mcp/               # Model Context Protocol
    ├── server.py      # MCP server
    ├── base.py        # Base classes
    ├── registry.py    # Tool registry
    └── tools/         # Tool implementations
        ├── file_tools.py   # File operations
        ├── search_tools.py # Search capabilities
        └── smart_tools.py  # Smart tools
```

### Key Components

1. **CopilotClient**: Main interface for AI-assisted code exploration
2. **LLMClient**: Unified interface for multiple LLM providers
3. **MCPServer**: Tool execution server following Model Context Protocol
4. **QueryOrchestrator**: Coordinates intent recognition and tool execution
5. **MemoryManager**: Manages conversation history and context

## Advanced Features

### Custom Personas

Create custom AI personas for specific use cases:

```python
from nighty_code.copilot.models.persona import CopilotPersona

custom_persona = CopilotPersona(
    name="reviewer",
    description="Code review specialist",
    traits=["thorough", "security-focused", "best-practices"],
    focus_areas=["code quality", "security vulnerabilities", "performance"]
)
```

### Session Management

Resume previous conversations:

```python
# Start a new session
copilot.chat()  # Note the session ID

# Later, resume the session
copilot.chat(resume_session="session_id_here")
```

### Tool Extension

Add custom tools to the MCP server:

```python
from nighty_code.mcp.base import MCPTool, ToolDefinition

@register_tool
class CustomTool(MCPTool):
    definition = ToolDefinition(
        name="custom_tool",
        description="My custom tool",
        category=ToolCategory.ANALYSIS
    )
    
    async def execute(self, params: Dict[str, Any]) -> Any:
        # Tool implementation
        return {"result": "success"}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_copilot.py
pytest tests/test_llm.py
pytest tests/test_mcp.py

# Run with coverage
pytest --cov=nighty_code tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking (if using mypy)
mypy src/
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Acknowledgments

Built with:
- Anthropic Claude API
- OpenAI API (optional)
- LangChain for memory management
- Pydantic for data validation
- Rich for terminal formatting