# MCP (Model Context Protocol) Module

A comprehensive Python implementation of the Model Context Protocol for seamless tool integration, filesystem operations, and server/client communication.

## Features

### 🛠️ Tool Management
- **Dynamic Tool Registration**: Register and discover tools at runtime
- **Tool Execution**: Safe execution with parameter validation
- **Tool Routing**: Intelligent routing to appropriate tools
- **Custom Tools**: Easy creation of custom tool implementations

### 📁 Filesystem Operations
- **File Management**: Read, write, move, delete files safely
- **Directory Operations**: List, create, search directories
- **Pattern Matching**: Glob and regex pattern support
- **Grep Search**: Fast code searching with ripgrep integration
- **Access Control**: Configurable directory permissions

### 🔌 Server/Client Architecture
- **Multiple Servers**: Support for various server types
- **Resource Discovery**: Automatic resource enumeration
- **Prompt Templates**: Reusable prompt management
- **Batch Operations**: Efficient bulk processing

## Installation

```bash
# Install required dependencies
pip install aiofiles pathlib typing-extensions
```

## Quick Start

```python
import asyncio
from pathlib import Path
from src.mcp import FilesystemServer, ToolCall

async def main():
    # Initialize filesystem server
    server = FilesystemServer(
        allowed_directories=[Path.cwd()],
        read_only=False
    )
    
    await server.initialize()
    
    # List available tools
    tools = await server.list_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Read a file
    result = await server.call_tool(ToolCall(
        name="read_file",
        arguments={"path": "README.md"}
    ))
    
    if result.is_success:
        print(f"File content: {result.content[0].text[:100]}...")
    
    await server.cleanup()

asyncio.run(main())
```

## Filesystem Operations

### Reading Files
```python
# Read text file
read_call = ToolCall(
    name="read_file",
    arguments={"path": "example.txt"}
)
result = await server.call_tool(read_call)

# Read with line limits
read_call = ToolCall(
    name="read_file",
    arguments={
        "path": "large_file.txt",
        "start_line": 100,
        "end_line": 200
    }
)
```

### Writing Files
```python
# Write new file
write_call = ToolCall(
    name="write_file",
    arguments={
        "path": "output.txt",
        "content": "Hello, MCP!"
    }
)

# Append to existing file
append_call = ToolCall(
    name="append_to_file",
    arguments={
        "path": "log.txt",
        "content": "New log entry\n"
    }
)
```

### Searching Files
```python
# Search by pattern
search_call = ToolCall(
    name="search_files",
    arguments={
        "pattern": "*.py",
        "directory": "src",
        "recursive": True
    }
)

# Grep for patterns
grep_call = ToolCall(
    name="grep",
    arguments={
        "pattern": "class \\w+",
        "directory": "src",
        "file_pattern": "*.py",
        "max_results": 20
    }
)
```

## Advanced Server Configuration

```python
from src.mcp import FilesystemServer
from pathlib import Path

server = FilesystemServer(
    # Access control
    allowed_directories=[
        Path.cwd() / "data",
        Path.cwd() / "docs"
    ],
    read_only=False,
    
    # File handling
    max_file_size=10 * 1024 * 1024,  # 10MB limit
    enable_hidden_files=False,
    
    # Features
    enable_grep=True,
    enable_diff=True,
    
    # Performance
    cache_enabled=True,
    cache_ttl=300  # 5 minutes
)
```

## Resource Management

```python
# List available resources
resources = await server.list_resources()
for resource in resources:
    print(f"{resource.name}: {resource.uri}")
    if resource.description:
        print(f"  {resource.description}")

# Read specific resource
content = await server.read_resource(resources[0].uri)
print(f"Resource content: {content}")
```

## Prompt Templates

```python
# List available prompts
prompts = await server.list_prompts()
for prompt in prompts:
    print(f"{prompt.name}: {prompt.description}")
    if prompt.arguments:
        print(f"  Args: {[arg.name for arg in prompt.arguments]}")

# Get prompt with arguments
result = await server.get_prompt(
    "code_review",
    arguments={
        "language": "python",
        "focus": "performance"
    }
)
```

## MCP Manager

Manage multiple servers simultaneously:

```python
from src.mcp import MCPManager, FilesystemServer

manager = MCPManager()

# Register multiple servers
fs_server = FilesystemServer(allowed_directories=[Path.cwd()])
await manager.register_server("filesystem", fs_server)

# Custom server example
# await manager.register_server("database", DatabaseServer())
# await manager.register_server("api", APIServer())

# Execute tools on specific servers
result = await manager.execute_tool(
    server_name="filesystem",
    tool_call=ToolCall(
        name="list_directory",
        arguments={"path": "."}
    )
)

# Discover all available tools
all_tools = await manager.discover_tools()
for server_name, tools in all_tools.items():
    print(f"{server_name}: {len(tools)} tools available")
```

## Creating Custom Tools

```python
from src.mcp.core import BaseTool, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="My custom tool",
            parameters={
                "input": {"type": "string", "required": True},
                "option": {"type": "boolean", "default": False}
            }
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        input_text = arguments.get("input")
        option = arguments.get("option", False)
        
        # Tool logic here
        result = f"Processed: {input_text} (option={option})"
        
        return ToolResult(
            success=True,
            content=[{"type": "text", "text": result}]
        )
```

## Batch Operations

```python
# Batch file operations
files_to_create = [
    ("file1.txt", "Content 1"),
    ("file2.txt", "Content 2"),
    ("file3.txt", "Content 3"),
]

for filepath, content in files_to_create:
    await server.call_tool(ToolCall(
        name="write_file",
        arguments={"path": filepath, "content": content}
    ))

# Batch read
contents = []
for filepath, _ in files_to_create:
    result = await server.call_tool(ToolCall(
        name="read_file",
        arguments={"path": filepath}
    ))
    if result.is_success:
        contents.append(result.content[0].text)
```

## Error Handling

```python
from src.mcp import MCPException, ToolNotFoundError, PermissionError

try:
    result = await server.call_tool(tool_call)
except ToolNotFoundError as e:
    print(f"Tool not found: {e}")
except PermissionError as e:
    print(f"Permission denied: {e}")
except MCPException as e:
    print(f"MCP error: {e}")
```

## Security Features

### Access Control
- Restrict operations to specific directories
- Read-only mode for safe exploration
- File size limits to prevent abuse
- Hidden file filtering

### Safe Execution
- Parameter validation
- Path traversal prevention
- Resource limits
- Timeout controls

## Module Structure

```
src/mcp/
├── __init__.py           # Main exports
├── core/
│   ├── base_client.py    # Base client implementation
│   ├── base_server.py    # Base server implementation
│   ├── exceptions.py     # Custom exceptions
│   ├── manager.py        # MCP manager
│   └── types.py          # Type definitions
├── servers/
│   └── filesystem.py     # Filesystem server
├── clients/              # Client implementations
└── utils/               # Utility functions
```

## Performance Optimization

1. **Enable Caching**: For frequently accessed files
2. **Use Batch Operations**: Process multiple items together
3. **Limit Search Scope**: Use specific directories and patterns
4. **Async Operations**: Leverage async/await for concurrency
5. **Resource Limits**: Set appropriate file size and result limits

## Common Use Cases

### Code Analysis
```python
# Find all Python classes
result = await server.call_tool(ToolCall(
    name="grep",
    arguments={
        "pattern": "^class \\w+",
        "directory": "src",
        "file_pattern": "*.py"
    }
))
```

### File Synchronization
```python
# Compare files
diff_result = await server.call_tool(ToolCall(
    name="diff_files",
    arguments={
        "file1": "version1.txt",
        "file2": "version2.txt",
        "context_lines": 3
    }
))
```

### Documentation Generation
```python
# List all markdown files
docs = await server.call_tool(ToolCall(
    name="search_files",
    arguments={
        "pattern": "*.md",
        "directory": "docs"
    }
))
```

## License

MIT License - See LICENSE file for details