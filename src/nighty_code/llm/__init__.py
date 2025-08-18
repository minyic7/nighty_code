"""
LLM integration module for nighty_code.

This module provides a unified interface for interacting with Large Language Models,
with support for:
- Multiple LLM providers (starting with OpenAI)
- Token counting and validation
- Automatic continuation for long outputs
- Response merging
- Error handling and retries
"""

from .client import LLMClient
from .config import LLMConfig, ModelConfig
from .exceptions import (
    LLMException,
    InputTooLargeError,
    OutputTruncatedError,
    TokenLimitExceededError
)

__all__ = [
    'LLMClient',
    'LLMConfig',
    'ModelConfig',
    'LLMException',
    'InputTooLargeError',
    'OutputTruncatedError',
    'TokenLimitExceededError'
]