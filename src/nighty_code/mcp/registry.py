"""
Tool registry for MCP server.
"""

from typing import Dict, List, Optional, Any, Type
import importlib
import inspect
from pathlib import Path

from .base import (
    MCPTool, ToolDefinition, ToolCategory,
    ToolNotFoundError, MCPException
)


class ToolRegistry:
    """Registry for managing MCP tools."""
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {}
        
    def register(self, tool: MCPTool) -> None:
        """Register a tool."""
        definition = tool.definition
        
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' already registered")
        
        self._tools[definition.name] = tool
        
        # Organize by category
        if definition.category not in self._categories:
            self._categories[definition.category] = []
        self._categories[definition.category].append(definition.name)
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            del self._tools[tool_name]
            
            # Remove from category
            category = tool.definition.category
            if category in self._categories:
                self._categories[category].remove(tool_name)
    
    def get_tool(self, name: str) -> MCPTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def list_enabled_tools(self) -> List[str]:
        """List enabled tool names."""
        return [
            name for name, tool in self._tools.items()
            if tool.definition.enabled
        ]
    
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get tools in a specific category."""
        return self._categories.get(category, [])
    
    def get_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Get all tool definitions."""
        return {
            name: tool.definition
            for name, tool in self._tools.items()
        }
    
    async def execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters."""
        tool = self.get_tool(name)
        
        if not tool.definition.enabled:
            raise MCPException(
                "TOOL_DISABLED",
                f"Tool '{name}' is currently disabled"
            )
        
        try:
            # Execute the tool
            result = await tool.execute(**params)
            return result
        except MCPException:
            raise
        except Exception as e:
            raise MCPException(
                "TOOL_EXECUTION_ERROR",
                f"Error executing tool '{name}': {str(e)}",
                {"tool": name, "error": str(e)}
            )
    
    def auto_discover(self, module_path: str) -> None:
        """Auto-discover and register tools from a module."""
        try:
            module = importlib.import_module(module_path)
            
            # Find all MCPTool implementations
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, MCPTool) and 
                    obj != MCPTool):
                    try:
                        # Instantiate and register
                        tool_instance = obj()
                        self.register(tool_instance)
                    except Exception as e:
                        print(f"Failed to register tool {name}: {e}")
                        
        except ImportError as e:
            print(f"Failed to import module {module_path}: {e}")


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tool(tool_class: Type[MCPTool]) -> Type[MCPTool]:
    """Decorator to register a tool class."""
    try:
        tool_instance = tool_class()
        _global_registry.register(tool_instance)
    except Exception as e:
        print(f"Failed to register tool {tool_class.__name__}: {e}")
    return tool_class