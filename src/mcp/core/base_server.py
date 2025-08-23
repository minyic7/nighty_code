# src/mcp/core/base_server.py
"""Base MCP server implementation"""

from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
import asyncio

from .types import ToolCall, ToolResult, TextContent, ErrorContent
from .exceptions import MCPToolNotFoundError, MCPServerError


logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerCapabilities:
    """MCP server capabilities"""
    tools: bool = True
    resources: bool = False
    prompts: bool = False
    logging: bool = False


class BaseMCPServer(ABC):
    """
    Base class for MCP servers.
    
    This provides both:
    1. Direct Python API for local use (no protocol overhead)
    2. MCP protocol server for remote/stdio communication
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, ToolDefinition] = {}
        self.capabilities = ServerCapabilities()
        self._mcp_server = None  # Reserved for future MCP protocol support
        self._initialized = False
        
    async def initialize(self):
        """Initialize the server"""
        if self._initialized:
            return
            
        # Register tools
        await self._register_tools()
        
        # Setup MCP server if needed
        if self.capabilities.tools:
            self._setup_mcp_server()
            
        self._initialized = True
        logger.info(f"MCP Server '{self.name}' initialized with {len(self.tools)} tools")
        
    @abstractmethod
    async def _register_tools(self):
        """Register all tools - must be implemented by subclasses"""
        pass
        
    def register_tool(self, tool: ToolDefinition):
        """Register a single tool"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
        
    def register_tool_decorator(self, 
                               name: Optional[str] = None,
                               description: Optional[str] = None,
                               input_schema: Optional[Dict[str, Any]] = None):
        """Decorator for registering tools"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "No description"
            schema = input_schema or {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            tool = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                input_schema=schema,
                handler=func
            )
            self.register_tool(tool)
            return func
        return decorator
        
    # Direct Python API (for local use)
    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Direct tool execution for local Python use.
        No MCP protocol overhead.
        """
        if tool_call.name not in self.tools:
            raise MCPToolNotFoundError(f"Tool '{tool_call.name}' not found")
            
        tool = self.tools[tool_call.name]
        
        try:
            if tool.handler:
                result = await tool.handler(**tool_call.arguments)
            else:
                result = await self._execute_tool(tool_call.name, tool_call.arguments)
                
            # Convert result to standard format
            if isinstance(result, str):
                content = [TextContent(text=result)]
            elif isinstance(result, list):
                content = result
            else:
                content = [TextContent(text=str(result))]
                
            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[ErrorContent(error=str(e))],
                status="error"
            )
            
    @abstractmethod
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool - can be overridden by subclasses"""
        pass
        
    def list_tools(self) -> List[ToolDefinition]:
        """List all available tools"""
        return list(self.tools.values())
        
    # MCP Protocol Support (Future Implementation)
    def _setup_mcp_server(self):
        """Setup MCP protocol server - reserved for future implementation"""
        # When MCP SDK is properly integrated, this will set up the protocol server
        # For now, we focus on the local Python API
        pass
            
    async def run_stdio(self):
        """Run as stdio MCP server - reserved for future implementation"""
        raise NotImplementedError(
            "Stdio MCP server not yet implemented. "
            "Use local Python API for now."
        )
            
    async def run_http(self, host: str = "localhost", port: int = 8000):
        """Run as HTTP MCP server - reserved for future implementation"""
        raise NotImplementedError(
            "HTTP MCP server not yet implemented. "
            "Use local Python API for now."
        )