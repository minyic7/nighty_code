# src/mcp/core/base_client.py
"""Base MCP client implementation"""

from typing import Dict, Any, List, Optional
from abc import ABC
import logging
from dataclasses import dataclass
import asyncio

from .types import ToolCall, ToolResult
from .exceptions import MCPConnectionError, MCPTimeoutError


logger = logging.getLogger(__name__)


@dataclass
class MCPClientConfig:
    """Configuration for MCP client"""
    name: str
    connection_type: str = "stdio"  # stdio, sse, or local
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    timeout: int = 30
    metadata: Dict[str, Any] = None


class BaseMCPClient(ABC):
    """
    Base class for MCP clients.
    
    Supports:
    1. Local server access (direct Python calls)
    2. Stdio server access (subprocess) - Future implementation
    3. SSE/HTTP server access (network) - Future implementation
    
    Note: The actual MCP protocol client implementation requires proper
    session management which is complex. For now, we focus on local server access.
    """
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.local_server = None  # For local server access
        self._connected = False
        self._tools: Dict[str, Any] = {}
        
    async def connect(self, local_server=None):
        """Connect to MCP server"""
        if self._connected:
            return
            
        try:
            if local_server:
                # Direct local connection
                self.local_server = local_server
                await self.local_server.initialize()
                self._connected = True
                logger.info(f"Connected to local server: {self.config.name}")
                
            elif self.config.connection_type == "stdio":
                # Stdio subprocess connection - requires complex session management
                # For production use, consider using the official MCP client libraries
                # or implementing a proper session manager
                raise NotImplementedError(
                    "Stdio connection requires complex session management. "
                    "Use local server mode or implement proper MCP client."
                )
                
            elif self.config.connection_type == "sse":
                # SSE/HTTP connection - not yet implemented in MCP SDK
                raise MCPConnectionError("SSE connection not yet supported")
                
            else:
                raise MCPConnectionError(f"Unknown connection type: {self.config.connection_type}")
                
            # Cache available tools
            await self._cache_tools()
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise MCPConnectionError(f"Failed to connect to {self.config.name}: {e}")
            
    async def _cache_tools(self):
        """Cache available tools from server"""
        tools = await self.list_tools()
        self._tools = {tool['name']: tool for tool in tools}
        
    async def disconnect(self):
        """Disconnect from MCP server"""
        if not self._connected:
            return
            
        try:
            self.local_server = None
            self._connected = False
            logger.info(f"Disconnected from server: {self.config.name}")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self._connected:
            raise MCPConnectionError("Not connected to server")
            
        try:
            if self.local_server:
                # Direct local call
                tools = self.local_server.list_tools()
                return [
                    {
                        'name': tool.name,
                        'description': tool.description,
                        'inputSchema': tool.input_schema
                    }
                    for tool in tools
                ]
            else:
                raise NotImplementedError("Remote server access not yet implemented")
                
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise
            
    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Call a tool on the server"""
        if not self._connected:
            raise MCPConnectionError("Not connected to server")
            
        try:
            if self.local_server:
                # Direct local call
                return await self.local_server.call_tool(tool_call)
            else:
                raise NotImplementedError("Remote server access not yet implemented")
                
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Tool call timed out: {tool_call.name}")
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            from .types import ErrorContent
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[ErrorContent(error=str(e))],
                status="error"
            )
            
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        return self._tools.get(tool_name)
        
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available"""
        return tool_name in self._tools
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()