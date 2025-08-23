# src/mcp/core/types.py
"""Core MCP types and data structures"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """MCP content types"""
    TEXT = "text"
    IMAGE = "image"
    ERROR = "error"


class ToolStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class TextContent:
    """Text content for MCP responses"""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content for MCP responses"""
    type: Literal["image"] = "image"
    data: str = ""  # Base64 encoded
    mime_type: str = "image/png"


@dataclass
class ErrorContent:
    """Error content for MCP responses"""
    type: Literal["error"] = "error"
    error: str = ""
    details: Optional[Dict[str, Any]] = None


ContentTypes = Union[TextContent, ImageContent, ErrorContent]


class ToolCall(BaseModel):
    """Request to call an MCP tool"""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    id: Optional[str] = Field(None, description="Unique call ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from an MCP tool execution"""
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call")
    content: List[ContentTypes] = Field(default_factory=list)
    status: ToolStatus = Field(ToolStatus.SUCCESS)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[float] = None


@dataclass
class MCPRequest:
    """Generic MCP request"""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Generic MCP response"""
    result: Any
    id: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)