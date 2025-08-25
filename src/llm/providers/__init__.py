"""LLM provider implementations."""

from typing import Type, Dict
from .base import BaseLLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .genai import GenAIProvider


# Registry of available providers
PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "genai": GenAIProvider,
}


def get_provider(provider_name: str) -> Type[BaseLLMProvider]:
    """Get provider class by name."""
    provider_class = PROVIDERS.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class


__all__ = [
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GenAIProvider",
    "get_provider",
    "PROVIDERS",
]