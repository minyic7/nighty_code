# Nighty Code

A comprehensive AI development framework featuring modular components for LLM interaction, tool orchestration, cognitive workflows, and intelligent data extraction.

## 🚀 Overview

Nighty Code is a production-ready Python framework that provides four powerful modules for building sophisticated AI applications:

- **LLM Module**: Multi-provider LLM client with structured output and middleware
- **MCP Module**: Model Context Protocol implementation for tool integration
- **Copilot Module**: Cognitive workflow orchestration with memory and reasoning
- **DataMiner Module**: Intelligent data extraction from unstructured content

## 📦 Modules

### [LLM Module](src/llm/README.md)
Unified interface for multiple LLM providers with advanced features:
- 🔄 Support for Anthropic Claude and OpenAI GPT
- 📊 Structured output with Pydantic/Instructor
- 🌊 Token streaming for real-time responses
- 🔁 Automatic retry with exponential backoff
- 📈 Metrics collection and monitoring
- 🔌 Connection pooling for high throughput

### [MCP Module](src/mcp/README.md)
Model Context Protocol implementation for seamless tool integration:
- 🛠️ Dynamic tool registration and discovery
- 📁 Safe filesystem operations with access control
- 🔍 Fast code searching with grep/ripgrep
- 📦 Resource and prompt template management
- 🔄 Batch operations support
- 🔒 Security features (path traversal prevention, limits)

### [Copilot Module](src/copilot/README.md)
Intelligent workflow orchestration with cognitive capabilities:
- 🧠 Multi-step reasoning and planning
- 💬 Interactive conversational sessions
- 🔧 Intelligent tool routing
- 📝 Short and long-term memory management
- 🛡️ Safety guardrails and content filtering
- 🤝 Multi-agent collaboration support

### [DataMiner Module](src/dataminer/README.md)
Advanced data extraction system using LLMs:
- 📊 Multiple extraction strategies (Simple, Multi-stage, Cognitive)
- 🎯 Type-safe extraction with Pydantic schemas
- 🔍 Repository-wide code analysis
- 📈 Confidence scoring and gap analysis
- ⚡ Batch processing with concurrency
- 💾 Intelligent result caching

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or poetry

### Install Dependencies

```bash
# Core dependencies
pip install pydantic asyncio typing-extensions

# LLM module dependencies
pip install anthropic openai instructor tiktoken

# Additional dependencies
pip install aiofiles numpy pathlib
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# API Keys
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional configurations
DATAMINER_MODE=fast
DATAMINER_CACHE_DIR=~/.dataminer_cache
```

## 🚀 Quick Start

### LLM Module Example

```python
import asyncio
from src.llm import LLMManager, Message, MessageRole, LLMProvider

async def main():
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(role=MessageRole.USER, content="Explain quantum computing")
    ]
    
    response = await client.complete(messages=messages)
    print(response.content)

asyncio.run(main())
```

### MCP Module Example

```python
from src.mcp import FilesystemServer, ToolCall
from pathlib import Path

server = FilesystemServer(allowed_directories=[Path.cwd()])
await server.initialize()

result = await server.call_tool(ToolCall(
    name="read_file",
    arguments={"path": "README.md"}
))
```

### Copilot Module Example

```python
from src.copilot import CopilotClient, InteractiveSession

client = CopilotClient()
await client.initialize()

session = InteractiveSession("demo", enable_memory=True)
response = await session.chat("Help me write a Python function")
print(response.content)
```

### DataMiner Module Example

```python
from src.dataminer import DataMinerClient
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

client = DataMinerClient()
await client.initialize()

text = "John Smith is a 30-year-old software engineer."
result = await client.extract(content=text, schema=PersonInfo)
print(f"Name: {result.data.name}, Age: {result.data.age}")
```

## 📚 Examples

Comprehensive examples for each module are available in the `examples/` directory:

- [`examples/llm_usage.py`](examples/llm_usage.py) - LLM module demonstrations
- [`examples/mcp_usage.py`](examples/mcp_usage.py) - MCP module demonstrations
- [`examples/copilot_usage.py`](examples/copilot_usage.py) - Copilot module demonstrations
- [`examples/dataminer_usage.py`](examples/dataminer_usage.py) - DataMiner module demonstrations

Run any example:
```bash
python examples/llm_usage.py
```

## 🏗️ Project Structure

```
nighty_code/
├── src/
│   ├── llm/              # LLM interaction module
│   │   ├── core/         # Core client and manager
│   │   ├── middleware/   # Retry, rate limit, metrics
│   │   └── providers/    # Provider implementations
│   ├── mcp/              # Model Context Protocol
│   │   ├── core/         # Base server/client
│   │   ├── servers/      # Server implementations
│   │   └── utils/        # Utility functions
│   ├── copilot/          # Workflow orchestration
│   │   ├── client/       # Interactive sessions
│   │   ├── core/         # Workflow engine
│   │   ├── guardrails/   # Safety checks
│   │   ├── memory/       # Memory management
│   │   └── nodes/        # Cognitive nodes
│   └── dataminer/        # Data extraction
│       ├── core/         # Configuration and types
│       ├── models/       # Extraction schemas
│       ├── strategies/   # Extraction strategies
│       └── utils/        # Validation and analysis
├── examples/             # Usage examples
├── tests/               # Test suites
├── config/              # Configuration files
└── docs/               # Documentation
```

## 🔧 Configuration

### Global Configuration

Create `config/settings.yaml`:

```yaml
llm:
  default_provider: anthropic
  retry_attempts: 3
  rate_limit: 60

mcp:
  allowed_directories:
    - ./data
    - ./workspace
  max_file_size: 10485760

copilot:
  enable_memory: true
  enable_guardrails: true
  session_timeout: 300

dataminer:
  default_mode: thorough
  cache_enabled: true
  confidence_threshold: 0.7
```

### Module-Specific Configuration

Each module can be configured independently:

```python
# LLM Configuration
from src.llm import LLMConfig
config = LLMConfig(
    retry_max_attempts=3,
    rate_limit_requests_per_minute=60
)

# DataMiner Configuration
from src.dataminer import DataMinerConfig
config = DataMinerConfig(
    extraction=ExtractionConfig(
        default_mode=ProcessingMode.THOROUGH,
        min_confidence_threshold=0.7
    )
)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_llm.py
python -m pytest tests/test_mcp.py
python -m pytest tests/test_copilot.py
python -m pytest tests/test_dataminer.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 🤝 Integration Examples

### Combining Modules

```python
# Use Copilot with DataMiner for intelligent extraction
from src.copilot import CopilotWorkflow
from src.dataminer import DataMinerClient

workflow = CopilotWorkflow()
dataminer = DataMinerClient()

# Copilot analyzes and DataMiner extracts
analysis = await workflow.execute([{
    "name": "analyze",
    "input": "Analyze this document structure"
}])

extraction = await dataminer.extract(
    content=analysis.output,
    schema=DocumentSchema
)
```

### Building an AI Assistant

```python
from src.llm import LLMManager
from src.mcp import FilesystemServer
from src.copilot import InteractiveSession

# Initialize components
llm = LLMManager()
filesystem = FilesystemServer()
session = InteractiveSession(
    enable_memory=True,
    enable_tools=True
)

# AI can now chat, use tools, and remember context
response = await session.chat("Read config.yaml and explain it")
```

## 📈 Performance Considerations

- **Connection Pooling**: Use connection pools for high-throughput scenarios
- **Caching**: Enable caching for repeated operations
- **Batch Processing**: Process multiple items concurrently
- **Async Operations**: Leverage async/await for non-blocking I/O
- **Resource Limits**: Set appropriate memory and rate limits

## 🛡️ Security

- **API Key Management**: Never commit API keys; use environment variables
- **Access Control**: Configure allowed directories for filesystem operations
- **Input Validation**: All inputs are validated before processing
- **Guardrails**: Content filtering and safety checks are available
- **Rate Limiting**: Prevent API abuse with configurable limits

## 📝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- OpenAI for GPT API
- Instructor library for structured output
- The open-source community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nighty_code/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nighty_code/discussions)
- **Email**: support@nightycode.com

## 🚦 Status

- ✅ LLM Module: Production Ready
- ✅ MCP Module: Production Ready
- ✅ Copilot Module: Beta
- ✅ DataMiner Module: Production Ready

---

Built with ❤️ by the Nighty Code team