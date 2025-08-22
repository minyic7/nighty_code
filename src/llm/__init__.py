"""
LLM Module - Connection pool and client management for LLM providers
"""

from .core.types import (
    LLMProvider,
    MessageRole,
    Message,
    LLMConfig,
    CompletionRequest,
    CompletionResponse,
    ClientStatus,
)
from .core.config import PoolConfig, ConfigManager, config_manager
from .core.exceptions import (
    LLMException,
    LLMProviderError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMPoolExhaustedError,
    LLMConfigurationError,
    LLMTimeoutError,
)
from .core.client import LLMClient
from .core.pool import LLMConnectionPool
from .core.manager import LLMManager, get_llm_manager

__all__ = [
    # Types
    "LLMProvider",
    "MessageRole",
    "Message",
    "LLMConfig",
    "CompletionRequest",
    "CompletionResponse",
    "ClientStatus",
    # Configuration
    "PoolConfig",
    "ConfigManager",
    "config_manager",
    # Exceptions
    "LLMException",
    "LLMProviderError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMPoolExhaustedError",
    "LLMConfigurationError",
    "LLMTimeoutError",
    # Core components
    "LLMClient",
    "LLMConnectionPool",
    "LLMManager",
    "get_llm_manager",
]