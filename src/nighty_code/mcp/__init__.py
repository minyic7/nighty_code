"""
Model Context Protocol (MCP) server for project exploration.
"""

from .server import MCPServer, run_server
from .registry import get_registry, register_tool
from .base import (
    MCPTool, ToolDefinition, ToolCategory, ToolParameter,
    MCPRequest, MCPResponse, MCPError, MCPException
)

__all__ = [
    "MCPServer",
    "run_server",
    "get_registry",
    "register_tool",
    "MCPTool",
    "ToolDefinition",
    "ToolCategory",
    "ToolParameter",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPException"
]