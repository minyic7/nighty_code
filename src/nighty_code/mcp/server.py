"""
MCP Server implementation.
"""

import asyncio
import json
from typing import Any, Dict, Optional
from pathlib import Path
import uuid

from .base import (
    MCPRequest, MCPResponse, MCPError, MCPException,
    ToolNotFoundError
)
from .registry import get_registry


class MCPServer:
    """Model Context Protocol server."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.registry = get_registry()
        
        # Auto-discover tools
        self._auto_discover_tools()
    
    def _auto_discover_tools(self):
        """Auto-discover and register tools."""
        # Import tools modules to trigger registration
        try:
            from . import tools
        except ImportError as e:
            print(f"Failed to import tools: {e}")
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request."""
        try:
            # Parse request
            request = MCPRequest(
                id=request_data.get("id", str(uuid.uuid4())),
                method=request_data.get("method", ""),
                params=request_data.get("params", {})
            )
            
            # Route request
            if request.method == "list_tools":
                result = await self._handle_list_tools()
            elif request.method == "describe_tool":
                result = await self._handle_describe_tool(request.params)
            elif request.method.startswith("tool/"):
                tool_name = request.method[5:]  # Remove "tool/" prefix
                result = await self._handle_tool_execution(tool_name, request.params)
            else:
                raise MCPException(
                    "METHOD_NOT_FOUND",
                    f"Unknown method: {request.method}"
                )
            
            # Build response
            response = MCPResponse(
                id=request.id,
                result=result
            )
            
            return self._serialize_response(response)
            
        except MCPException as e:
            # Handle MCP exceptions
            error_response = MCPResponse(
                id=request_data.get("id", "unknown"),
                error=e.to_dict() if hasattr(e, 'to_dict') else {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details
                }
            )
            return self._serialize_response(error_response)
            
        except Exception as e:
            # Handle unexpected exceptions
            error_response = MCPResponse(
                id=request_data.get("id", "unknown"),
                error={
                    "code": "INTERNAL_ERROR",
                    "message": f"Internal server error: {str(e)}"
                }
            )
            return self._serialize_response(error_response)
    
    async def _handle_list_tools(self) -> Dict[str, Any]:
        """Handle list_tools request."""
        tools = self.registry.get_tool_definitions()
        return {
            "tools": [
                {
                    "name": name,
                    "description": definition.description,
                    "category": definition.category.value,
                    "enabled": definition.enabled
                }
                for name, definition in tools.items()
            ]
        }
    
    async def _handle_describe_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle describe_tool request."""
        tool_name = params.get("name")
        if not tool_name:
            raise MCPException("INVALID_PARAMS", "Missing tool name")
        
        tool = self.registry.get_tool(tool_name)
        definition = tool.definition
        
        return {
            "name": definition.name,
            "description": definition.description,
            "category": definition.category.value,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default
                }
                for param in definition.parameters
            ],
            "returns": definition.returns,
            "examples": definition.examples,
            "version": definition.version,
            "enabled": definition.enabled
        }
    
    async def _handle_tool_execution(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Handle tool execution request."""
        result = await self.registry.execute_tool(tool_name, params)
        return result
    
    def _serialize_response(self, response: MCPResponse) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        result = {"id": response.id}
        
        if response.error:
            result["error"] = response.error
        else:
            result["result"] = response.result
        
        return result
    
    async def serve_stdio(self):
        """Serve MCP over stdio (for testing)."""
        import sys
        
        print("MCP Server started. Listening for requests...")
        print(f"Available tools: {', '.join(self.registry.list_tools())}")
        print("-" * 50)
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                
                # Parse JSON request
                request_data = json.loads(line.decode())
                
                # Handle request
                response = await self.handle_request(request_data)
                
                # Send response
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                error = {
                    "error": {
                        "code": "PARSE_ERROR",
                        "message": f"Invalid JSON: {str(e)}"
                    }
                }
                print(json.dumps(error))
                sys.stdout.flush()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Server error: {e}", file=sys.stderr)


async def run_server(project_root: Optional[Path] = None):
    """Run the MCP server."""
    server = MCPServer(project_root)
    await server.serve_stdio()


def main():
    """Main entry point."""
    import sys
    
    # Get project root from command line or use current directory
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    # Run server
    asyncio.run(run_server(project_root))


if __name__ == "__main__":
    main()