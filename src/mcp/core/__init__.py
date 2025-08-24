# src/mcp/core/__init__.py
"""Core MCP abstractions and base classes"""

from .base_server import BaseMCPServer, ToolDefinition, ServerCapabilities
from .base_client import BaseMCPClient, MCPClientConfig
from .manager import MCPManager
from .types import (
    MCPRequest,
    MCPResponse,
    ToolCall,
    ToolResult,
    TextContent,
    ImageContent,
    ErrorContent,
)
from .exceptions import (
    MCPException,
    MCPConnectionError,
    MCPToolNotFoundError,
    MCPServerError,
    MCPTimeoutError,
)

__all__ = [
    # Base classes
    "BaseMCPServer",
    "BaseMCPClient",
    "MCPManager",
    "ToolDefinition",
    "ServerCapabilities",
    "MCPClientConfig",
    # Types
    "MCPRequest",
    "MCPResponse",
    "ToolCall",
    "ToolResult",
    "TextContent",
    "ImageContent",
    "ErrorContent",
    # Exceptions
    "MCPException",
    "MCPConnectionError",
    "MCPToolNotFoundError",
    "MCPServerError",
    "MCPTimeoutError",
]