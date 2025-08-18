"""
Main LLM client implementation with continuation and merging support.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Generator
from dataclasses import dataclass, field

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .config import LLMConfig, ModelConfig, LLMProvider, TokenStrategy
from .exceptions import (
    InputTooLargeError,
    OutputTruncatedError,
    TokenLimitExceededError,
    APIError,
    ConfigurationError
)
from .token_utils import TokenCounter, ResponseMerger


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response with metadata."""
    
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    truncated: bool = False
    continued: bool = False
    chunks: List[str] = field(default_factory=list)
    total_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    
    def __str__(self) -> str:
        return self.content


class LLMClient:
    """
    Unified LLM client with automatic continuation and response merging.
    
    Features:
    - Input validation with token counting
    - Automatic output continuation when truncated
    - Response merging for multi-part responses
    - Cost tracking and warnings
    - Retry logic with exponential backoff
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize LLM client.
        
        Args:
            config: LLMConfig object or None to use defaults
            **kwargs: Override config parameters
        """
        
        # Create or update config
        if config is None:
            config = LLMConfig(**kwargs)
        else:
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        self.model_config = config.model_config
        
        # Initialize provider client
        self._init_provider_client()
        
        # Token counter and merger
        self.token_counter = TokenCounter()
        self.response_merger = ResponseMerger()
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
    
    def _init_provider_client(self):
        """Initialize the provider-specific client."""
        
        if self.config.provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ConfigurationError("OpenAI package not installed. Run: pip install openai")
            
            if not self.config.api_key:
                raise ConfigurationError("OpenAI API key not provided")
            
            # Initialize OpenAI client
            openai.api_key = self.config.api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base
            if self.config.organization:
                openai.organization = self.config.organization
            
            self.client = openai
            
        else:
            raise ConfigurationError(f"Provider {self.config.provider} not yet implemented")
    
    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with automatic continuation if needed.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate (uses model default if None)
            temperature: Temperature for generation
            system_prompt: System prompt (for chat models)
            **kwargs: Additional model parameters
            
        Returns:
            LLMResponse with complete response
        """
        
        # Prepare messages for chat models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Validate input tokens
        input_tokens = self.token_counter.count_messages_tokens(messages, self.config.model)
        
        if not self.model_config.validate_input_tokens(input_tokens):
            if self.config.input_token_strategy == TokenStrategy.ERROR:
                raise InputTooLargeError(input_tokens, self.model_config.max_input_tokens)
            elif self.config.input_token_strategy == TokenStrategy.TRUNCATE:
                # Truncate prompt to fit
                max_prompt_tokens = self.model_config.max_input_tokens - 100  # Leave room for system
                prompt = self.token_counter.truncate_text(
                    prompt, 
                    max_prompt_tokens, 
                    self.config.model
                )
                messages[-1]["content"] = prompt
                input_tokens = self.token_counter.count_messages_tokens(messages, self.config.model)
        
        # Calculate max output tokens
        if max_tokens is None:
            max_tokens = self.model_config.calculate_max_output_tokens(input_tokens)
        else:
            max_tokens = min(max_tokens, self.model_config.calculate_max_output_tokens(input_tokens))
        
        # Log request if enabled
        if self.config.log_requests:
            logger.info(f"LLM Request - Model: {self.config.model}, Input tokens: {input_tokens}, Max output: {max_tokens}")
        
        # Generate response with continuation support
        response_chunks = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_cost = 0.0
        continued = False
        
        current_messages = messages.copy()
        remaining_tokens = max_tokens
        max_continuation_attempts = 5  # Prevent infinite loops
        continuation_attempts = 0
        
        while continuation_attempts < max_continuation_attempts:
            try:
                # Make API call
                response = self._call_api(
                    messages=current_messages,
                    max_tokens=min(remaining_tokens, self.model_config.max_output_tokens),
                    temperature=temperature or self.model_config.temperature,
                    **kwargs
                )
                
                # Extract content and usage
                content = self._extract_content(response)
                usage = self._extract_usage(response)
                
                # Update totals
                response_chunks.append(content)
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                
                # Calculate cost
                if self.config.track_costs:
                    cost = self._calculate_cost(usage)
                    total_cost += cost
                    if cost > self.config.cost_warning_threshold:
                        logger.warning(f"High cost for single request: ${cost:.4f}")
                
                # Check if response was truncated
                truncated = self.response_merger.detect_truncation(content)
                
                if truncated and self.config.output_token_strategy == TokenStrategy.CONTINUE:
                    # Prepare for continuation
                    continuation_attempts += 1
                    continued = True
                    
                    # Create continuation prompt
                    merged_so_far = self.response_merger.merge_text_responses(response_chunks)
                    continuation_prompt = self.response_merger.create_continuation_prompt(
                        prompt,
                        merged_so_far
                    )
                    
                    # Update messages for continuation
                    current_messages = [
                        {"role": "user", "content": continuation_prompt}
                    ]
                    
                    # Update remaining tokens
                    remaining_tokens -= usage.get("completion_tokens", 0)
                    
                    if remaining_tokens <= 0:
                        logger.warning("Reached token limit, stopping continuation")
                        break
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.5)
                    
                else:
                    # Response complete
                    break
                    
            except Exception as e:
                logger.error(f"API call failed: {e}")
                if continuation_attempts == 0:
                    # First attempt failed, propagate error
                    raise
                else:
                    # Continuation failed, return what we have
                    break
        
        # Merge all response chunks
        final_content = self.response_merger.merge_text_responses(response_chunks)
        
        # Log response if enabled
        if self.config.log_responses:
            logger.info(f"LLM Response - Tokens: {total_usage['completion_tokens']}, Cost: ${total_cost:.4f}")
        
        # Update totals
        self.total_cost += total_cost
        self.request_count += 1
        
        return LLMResponse(
            content=final_content,
            model=self.config.model,
            usage=total_usage,
            cost=total_cost,
            truncated=truncated and not continued,
            continued=continued,
            chunks=response_chunks if continued else [],
            total_tokens=total_usage["total_tokens"],
            completion_tokens=total_usage["completion_tokens"],
            prompt_tokens=total_usage["prompt_tokens"]
        )
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Any:
        """
        Make actual API call with retry logic.
        
        Args:
            messages: Messages to send
            max_tokens: Max tokens to generate
            temperature: Temperature setting
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        
        last_error = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider == LLMProvider.OPENAI:
                    # OpenAI chat completion
                    response = self.client.ChatCompletion.create(
                        model=self.config.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=kwargs.get("top_p", self.model_config.top_p),
                        frequency_penalty=kwargs.get("frequency_penalty", self.model_config.frequency_penalty),
                        presence_penalty=kwargs.get("presence_penalty", self.model_config.presence_penalty),
                        **{k: v for k, v in kwargs.items() if k not in ["top_p", "frequency_penalty", "presence_penalty"]}
                    )
                    return response
                    
            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(delay)
                    if self.config.exponential_backoff:
                        delay *= 2
        
        # All retries failed
        raise APIError(self.config.provider.value, message=str(last_error))
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from API response."""
        
        if self.config.provider == LLMProvider.OPENAI:
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
        
        return ""
    
    def _extract_usage(self, response: Any) -> Dict[str, int]:
        """Extract usage information from API response."""
        
        if self.config.provider == LLMProvider.OPENAI:
            if hasattr(response, "usage"):
                return {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
        
        return {}
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost based on usage."""
        
        if not self.model_config:
            return 0.0
        
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * self.model_config.input_token_cost
        output_cost = (usage.get("completion_tokens", 0) / 1000) * self.model_config.output_token_cost
        
        return input_cost + output_cost
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count_tokens(text, self.config.model)
    
    def estimate_cost(self, prompt: str, expected_output_tokens: int = 1000) -> float:
        """Estimate cost for a prompt."""
        
        messages = [{"role": "user", "content": prompt}]
        
        return self.token_counter.estimate_messages_cost(
            messages,
            self.config.model,
            self.model_config.input_token_cost,
            expected_output_tokens,
            self.model_config.output_token_cost
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        
        return {
            "total_cost": self.total_cost,
            "request_count": self.request_count,
            "average_cost": self.total_cost / self.request_count if self.request_count > 0 else 0,
            "model": self.config.model,
            "provider": self.config.provider.value
        }
    
    @classmethod
    def create(cls, api_key: str, model: str = "gpt-3.5-turbo", **kwargs) -> "LLMClient":
        """
        Convenience method to create a client.
        
        Args:
            api_key: API key
            model: Model name
            **kwargs: Additional config parameters
            
        Returns:
            Configured LLMClient
        """
        
        config = LLMConfig(
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        return cls(config)