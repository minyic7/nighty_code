# src/mcp/core/manager.py
"""
MCP Manager for handling multiple MCP servers
"""

import logging
from typing import Dict, Optional, Any

from .base_server import BaseMCPServer
from .exceptions import MCPException

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages multiple MCP servers"""
    
    def __init__(self):
        self._servers: Dict[str, BaseMCPServer] = {}
        self._initialized = False
    
    def register_server(self, name: str, server: BaseMCPServer):
        """Register an MCP server"""
        if name in self._servers:
            logger.warning(f"Server '{name}' already registered, replacing")
        
        self._servers[name] = server
        logger.info(f"Registered MCP server: {name}")
    
    def unregister_server(self, name: str):
        """Unregister an MCP server"""
        if name in self._servers:
            del self._servers[name]
            logger.info(f"Unregistered MCP server: {name}")
    
    def get_server(self, name: str) -> Optional[BaseMCPServer]:
        """Get a specific server by name"""
        return self._servers.get(name)
    
    def get_servers(self) -> Dict[str, BaseMCPServer]:
        """Get all registered servers"""
        return self._servers.copy()
    
    async def call_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on a specific server"""
        from .types import ToolCall
        
        server = self.get_server(server_name)
        if not server:
            raise MCPException(f"Server '{server_name}' not found")
        
        tool_call = ToolCall(
            name=tool_name,
            arguments=arguments
        )
        return await server.call_tool(tool_call)
    
    async def list_all_tools(self) -> Dict[str, list]:
        """List all tools from all servers"""
        all_tools = {}
        
        for server_name, server in self._servers.items():
            try:
                tools = server.list_tools()
                all_tools[server_name] = tools
            except Exception as e:
                logger.error(f"Error listing tools from {server_name}: {e}")
                all_tools[server_name] = []
        
        return all_tools
    
    async def close_all(self):
        """Close all servers"""
        for server_name, server in self._servers.items():
            try:
                await server.close()
                logger.info(f"Closed server: {server_name}")
            except Exception as e:
                logger.error(f"Error closing server {server_name}: {e}")