# src/llm/core/types.py
from typing import Dict, Any, Optional, Union, List, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENAI = "genai"
    # Add more providers as needed


class MessageRole(str, Enum):
    """Message roles for chat completions"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message"""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)
    rate_limit_config: Optional[Any] = None  # Will be RateLimitConfig


@dataclass
class CompletionRequest:
    """Request for LLM completion"""
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from LLM completion"""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None  # Track which client handled the request


@dataclass
class ClientStatus:
    """Status of an LLM client"""
    client_id: str
    provider: LLMProvider
    model: str
    is_available: bool
    in_use: bool
    last_used: Optional[datetime] = None
    error_count: int = 0
    total_requests: int = 0