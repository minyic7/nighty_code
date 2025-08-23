# MCP (Model Context Protocol) Module

A clean, production-ready module for implementing MCP tools and servers. Designed for local Python API usage with future support for MCP protocol communication.

## Architecture Overview

```
src/mcp/
├── core/           # Base abstractions and types
│   ├── base_server.py    # Server base class
│   ├── base_client.py    # Client base class
│   ├── types.py          # Data structures
│   └── exceptions.py     # Custom exceptions
├── servers/        # MCP server implementations
│   └── filesystem.py     # File operations server
├── clients/        # Future: MCP client implementations
└── utils/          # Future: Utility functions
```

## Design Principles

1. **Clean Separation**: MCP module only handles tools, not LLM orchestration
2. **Local-First**: Optimized for direct Python API usage (no protocol overhead)
3. **Type-Safe**: Strong typing with dataclasses and Pydantic
4. **Secure**: Built-in path validation and size limits
5. **Extensible**: Easy to add new servers and tools

## Core Components

### BaseMCPServer
- Abstract base for all MCP servers
- Provides tool registration and execution
- Supports both decorator and direct registration

### FilesystemServer
- Secure file operations with path validation
- Tools: `read_file`, `list_directory`, `search_pattern`, `search_keyword`
- Configurable root path and file size limits

## Usage Examples

### 1. Basic File Operations

```python
from mcp import FilesystemServer, ToolCall

# Initialize server with project root
fs_server = FilesystemServer(root_path="/path/to/project")
await fs_server.initialize()

# Read a file
result = await fs_server.call_tool(
    ToolCall(
        name="read_file",
        arguments={"path": "src/main.py"}
    )
)
print(result.content[0].text)

# List directory contents
result = await fs_server.call_tool(
    ToolCall(
        name="list_directory",
        arguments={"path": "src", "recursive": True}
    )
)
print(result.content[0].text)
```

### 2. Search Operations

```python
# Search for Python files
result = await fs_server.call_tool(
    ToolCall(
        name="search_pattern",
        arguments={"pattern": "**/*.py", "path": "src"}
    )
)

# Search for keyword in files
result = await fs_server.call_tool(
    ToolCall(
        name="search_keyword",
        arguments={
            "keyword": "async def",
            "path": "src",
            "extensions": [".py"],
            "max_results": 50
        }
    )
)
```

### 3. Creating Custom MCP Server

```python
from mcp import BaseMCPServer, ToolDefinition, TextContent

class DatabaseServer(BaseMCPServer):
    def __init__(self, connection_string: str):
        super().__init__("database-server", "1.0.0")
        self.connection_string = connection_string
    
    async def _register_tools(self):
        self.register_tool(ToolDefinition(
            name="query",
            description="Execute SQL query",
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            },
            handler=self._execute_query
        ))
    
    async def _execute_query(self, sql: str):
        # Your database logic here
        result = await self.db.execute(sql)
        return TextContent(text=str(result))
```

### 4. Integration with Copilot/Application

```python
# In your copilot module (not in MCP)
from mcp import FilesystemServer, ToolCall
from llm import get_llm_manager, Message, MessageRole

class CopilotOrchestrator:
    def __init__(self):
        self.fs_server = FilesystemServer()
        self.llm_manager = None
        
    async def initialize(self):
        await self.fs_server.initialize()
        self.llm_manager = await get_llm_manager()
    
    async def process_with_tools(self, user_input: str):
        # Get available tools
        tools = self.fs_server.list_tools()
        
        # Format for LLM
        tool_descriptions = self._format_tools_for_llm(tools)
        
        # Call LLM
        client = self.llm_manager.get_client()
        messages = [
            Message(MessageRole.SYSTEM, f"You have these tools: {tool_descriptions}"),
            Message(MessageRole.USER, user_input)
        ]
        response = await client.complete(messages)
        
        # Parse and execute tool calls
        if self._has_tool_call(response.content):
            tool_call = self._parse_tool_call(response.content)
            result = await self.fs_server.call_tool(tool_call)
            # Continue conversation with result...
```

## Security Features

1. **Path Validation**: All paths are validated to stay within root directory
2. **Size Limits**: Configurable maximum file size (default: 10MB)
3. **Extension Filtering**: Optional allowed extensions list
4. **Error Handling**: Comprehensive error messages without exposing system details

## Error Handling

```python
from mcp import MCPValidationError, MCPToolNotFoundError

try:
    result = await fs_server.call_tool(tool_call)
except MCPValidationError as e:
    print(f"Invalid input: {e}")
except MCPToolNotFoundError as e:
    print(f"Tool not found: {e}")
```

## Available Tools

### FilesystemServer Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path`, `encoding` |
| `list_directory` | List directory contents | `path`, `recursive`, `include_hidden` |
| `search_pattern` | Find files by glob pattern | `pattern`, `path` |
| `search_keyword` | Search text in files | `keyword`, `path`, `extensions`, `use_regex`, `max_results` |

## Best Practices

1. **Initialize Once**: Create servers at application startup
2. **Use Type Hints**: Leverage the provided types for safety
3. **Handle Errors**: Always wrap tool calls in try-except
4. **Validate Input**: Use the built-in validation features
5. **Keep It Simple**: MCP handles tools, your app handles orchestration

## Future Enhancements

- [ ] Remote MCP server support (stdio/SSE)
- [ ] Additional servers (Database, API, Shell)
- [ ] Context-aware search
- [ ] File write operations (with safety checks)
- [ ] Caching layer for frequently accessed files

## Testing

```python
# Example test
import pytest
from mcp import FilesystemServer, ToolCall

@pytest.mark.asyncio
async def test_read_file():
    server = FilesystemServer(root_path="./test_data")
    await server.initialize()
    
    result = await server.call_tool(
        ToolCall(name="read_file", arguments={"path": "test.txt"})
    )
    
    assert result.status == "success"
    assert "expected content" in result.content[0].text
```

## License

Part of the nighty_code project.