"""
Configuration management for LLM clients.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum
import os


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"  # For local models


class TokenStrategy(Enum):
    """Strategies for handling token limits."""
    ERROR = "error"  # Raise error if limit exceeded
    TRUNCATE = "truncate"  # Truncate input to fit
    CHUNK = "chunk"  # Split into chunks (for input)
    CONTINUE = "continue"  # Continue generation (for output)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    
    name: str  # Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    max_input_tokens: int  # Maximum input tokens
    max_output_tokens: int  # Maximum output tokens  
    max_total_tokens: int  # Maximum total tokens (input + output)
    
    # Token costs (optional, for tracking)
    input_token_cost: float = 0.0  # Cost per 1K input tokens
    output_token_cost: float = 0.0  # Cost per 1K output tokens
    
    # Model capabilities
    supports_functions: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    
    # Performance settings
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def validate_input_tokens(self, token_count: int) -> bool:
        """Check if input token count is within limits."""
        return token_count <= self.max_input_tokens
    
    def calculate_max_output_tokens(self, input_tokens: int) -> int:
        """Calculate maximum output tokens given input tokens."""
        return min(
            self.max_output_tokens,
            self.max_total_tokens - input_tokens
        )


# Predefined model configurations
MODELS = {
    "gpt-4": ModelConfig(
        name="gpt-4",
        max_input_tokens=8192,
        max_output_tokens=4096,
        max_total_tokens=8192,
        input_token_cost=0.03,
        output_token_cost=0.06,
        supports_functions=True,
        supports_vision=False,
        supports_json_mode=True
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo-preview",
        max_input_tokens=128000,
        max_output_tokens=4096,
        max_total_tokens=128000,
        input_token_cost=0.01,
        output_token_cost=0.03,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        max_input_tokens=16385,
        max_output_tokens=4096,
        max_total_tokens=16385,
        input_token_cost=0.0005,
        output_token_cost=0.0015,
        supports_functions=True,
        supports_vision=False,
        supports_json_mode=True
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        name="gpt-3.5-turbo-16k",
        max_input_tokens=16385,
        max_output_tokens=4096,
        max_total_tokens=16385,
        input_token_cost=0.003,
        output_token_cost=0.004,
        supports_functions=True,
        supports_vision=False,
        supports_json_mode=True
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus-20240229",
        max_input_tokens=200000,
        max_output_tokens=4096,
        max_total_tokens=200000,
        input_token_cost=0.015,
        output_token_cost=0.075,
        supports_functions=False,
        supports_vision=True,
        supports_json_mode=False
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet-20240229",
        max_input_tokens=200000,
        max_output_tokens=4096,
        max_total_tokens=200000,
        input_token_cost=0.003,
        output_token_cost=0.015,
        supports_functions=False,
        supports_vision=True,
        supports_json_mode=False
    )
}


@dataclass
class LLMConfig:
    """Main configuration for LLM client."""
    
    # Provider settings
    provider: LLMProvider = LLMProvider.OPENAI
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For custom endpoints
    api_version: Optional[str] = None  # For Azure OpenAI
    organization: Optional[str] = None  # For OpenAI org ID
    
    # Model settings
    model: str = "gpt-3.5-turbo"
    model_config: Optional[ModelConfig] = None
    
    # Token handling
    input_token_strategy: TokenStrategy = TokenStrategy.ERROR
    output_token_strategy: TokenStrategy = TokenStrategy.CONTINUE
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Timeout settings
    request_timeout: float = 60.0
    
    # Logging and debugging
    log_requests: bool = False
    log_responses: bool = False
    log_tokens: bool = True
    
    # Cost tracking
    track_costs: bool = True
    cost_warning_threshold: float = 1.0  # Warn if single request costs > $1
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        
        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Load model configuration if not provided
        if self.model_config is None and self.model in MODELS:
            self.model_config = MODELS[self.model]
        elif self.model_config is None:
            # Create default config for unknown models
            self.model_config = ModelConfig(
                name=self.model,
                max_input_tokens=4096,
                max_output_tokens=2048,
                max_total_tokens=4096
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create configuration from dictionary."""
        
        # Handle provider enum
        if "provider" in config_dict and isinstance(config_dict["provider"], str):
            config_dict["provider"] = LLMProvider(config_dict["provider"])
        
        # Handle token strategies
        if "input_token_strategy" in config_dict and isinstance(config_dict["input_token_strategy"], str):
            config_dict["input_token_strategy"] = TokenStrategy(config_dict["input_token_strategy"])
        
        if "output_token_strategy" in config_dict and isinstance(config_dict["output_token_strategy"], str):
            config_dict["output_token_strategy"] = TokenStrategy(config_dict["output_token_strategy"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
        result = {
            "provider": self.provider.value,
            "model": self.model,
            "input_token_strategy": self.input_token_strategy.value,
            "output_token_strategy": self.output_token_strategy.value,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "exponential_backoff": self.exponential_backoff,
            "request_timeout": self.request_timeout,
            "log_requests": self.log_requests,
            "log_responses": self.log_responses,
            "log_tokens": self.log_tokens,
            "track_costs": self.track_costs,
            "cost_warning_threshold": self.cost_warning_threshold
        }
        
        # Add optional fields if present
        if self.api_base:
            result["api_base"] = self.api_base
        if self.api_version:
            result["api_version"] = self.api_version
        if self.organization:
            result["organization"] = self.organization
        
        return result