# src/llm/core/exceptions.py
from typing import Optional, Any


class LLMException(Exception):
    """Base exception for LLM module"""
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.details = details


class LLMProviderError(LLMException):
    """Raised when there's an error with the LLM provider"""
    pass


class LLMConnectionError(LLMException):
    """Raised when connection to LLM provider fails"""
    pass


class LLMRateLimitError(LLMException):
    """Raised when rate limit is exceeded"""
    pass


class LLMAuthenticationError(LLMException):
    """Raised when authentication fails"""
    pass


class LLMPoolExhaustedError(LLMException):
    """Raised when no clients are available in the pool"""
    pass


class LLMConfigurationError(LLMException):
    """Raised when there's a configuration error"""
    pass


class LLMTimeoutError(LLMException):
    """Raised when a request times out"""
    pass