"""
MCP (Model Context Protocol) Module

Provides MCP servers and tools for external tool integration.
This module focuses on tool implementation, not LLM orchestration.
"""

# Core components
from .core import (
    # Base classes
    BaseMCPServer,
    BaseMCPClient,
    # Types
    ToolDefinition,
    ServerCapabilities,
    MCPClientConfig,
    ToolCall,
    ToolResult,
    TextContent,
    ImageContent,
    ErrorContent,
    MCPRequest,
    MCPResponse,
    # Exceptions
    MCPException,
    MCPConnectionError,
    MCPToolNotFoundError,
    MCPServerError,
    MCPTimeoutError,
)

# Servers
from .servers import (
    FilesystemServer,
)

__all__ = [
    # Core - Base classes
    "BaseMCPServer",
    "BaseMCPClient",
    "ToolDefinition",
    "ServerCapabilities",
    "MCPClientConfig",
    # Core - Types
    "ToolCall",
    "ToolResult",
    "TextContent",
    "ImageContent",
    "ErrorContent",
    "MCPRequest",
    "MCPResponse",
    # Core - Exceptions
    "MCPException",
    "MCPConnectionError",
    "MCPToolNotFoundError",
    "MCPServerError",
    "MCPTimeoutError",
    # Servers
    "FilesystemServer",
]

__version__ = "1.0.0"