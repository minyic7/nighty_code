# src/mcp/core/exceptions.py
"""MCP-specific exceptions"""


class MCPException(Exception):
    """Base exception for MCP module"""
    pass


class MCPConnectionError(MCPException):
    """Connection-related errors"""
    pass


class MCPToolNotFoundError(MCPException):
    """Tool not found error"""
    pass


class MCPServerError(MCPException):
    """Server-side error"""
    pass


class MCPTimeoutError(MCPException):
    """Operation timeout error"""
    pass


class MCPValidationError(MCPException):
    """Input validation error"""
    pass