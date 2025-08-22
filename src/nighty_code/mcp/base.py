"""
Base classes and types for MCP server.
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Protocol
from dataclasses import dataclass, field
from enum import Enum
import inspect
from pathlib import Path


class ToolCategory(Enum):
    """Categories for organizing tools."""
    FILE = "file"
    SEARCH = "search"
    ANALYSIS = "analysis"
    NAVIGATION = "navigation"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of an MCP tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = "string"
    examples: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    enabled: bool = True


class MCPTool(Protocol):
    """Protocol for MCP tools."""
    
    @property
    def definition(self) -> ToolDefinition:
        """Tool definition."""
        ...
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...


@dataclass
class MCPRequest:
    """Incoming request to MCP server."""
    id: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Response from MCP server."""
    id: str
    result: Any = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPError:
    """Error response details."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response."""
        result = {
            "code": self.code,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


class MCPException(Exception):
    """Base exception for MCP errors."""
    def __init__(self, code: str, message: str, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class ToolNotFoundError(MCPException):
    """Tool not found error."""
    def __init__(self, tool_name: str):
        super().__init__(
            "TOOL_NOT_FOUND",
            f"Tool '{tool_name}' not found",
            {"tool": tool_name}
        )


class InvalidParameterError(MCPException):
    """Invalid parameter error."""
    def __init__(self, param_name: str, reason: str):
        super().__init__(
            "INVALID_PARAMETER",
            f"Invalid parameter '{param_name}': {reason}",
            {"parameter": param_name, "reason": reason}
        )


class SecurityError(MCPException):
    """Security violation error."""
    def __init__(self, message: str):
        super().__init__(
            "SECURITY_ERROR",
            message,
            None
        )