# src/llm/middleware/token_calculator.py
"""
Token calculation using actual tokenizers for different LLM providers.
"""

import logging
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google" 
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class TokenCalculator(ABC):
    """Abstract base class for token calculators"""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in a list of messages"""
        pass
    
    @abstractmethod
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        pass


class OpenAITokenCalculator(TokenCalculator):
    """Token calculator for OpenAI models using tiktoken"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self._encoding = None
        self._load_encoding()
    
    def _load_encoding(self):
        """Load the appropriate encoding for the model"""
        try:
            import tiktoken
            
            # Map model names to encodings
            model_encodings = {
                "gpt-4": "cl100k_base",
                "gpt-4-turbo": "cl100k_base", 
                "gpt-4o": "o200k_base",
                "gpt-3.5-turbo": "cl100k_base",
                "text-davinci-003": "p50k_base",
                "text-davinci-002": "p50k_base",
                "code-davinci-002": "p50k_base",
            }
            
            # Try to get encoding for specific model
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to mapping or default
                encoding_name = model_encodings.get(self.model_name, "cl100k_base")
                self._encoding = tiktoken.get_encoding(encoding_name)
                logger.info(f"Using fallback encoding {encoding_name} for model {self.model_name}")
                
        except ImportError:
            logger.warning("tiktoken not installed, falling back to rough estimation")
            self._encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self._encoding is None:
            # Fallback to rough estimation
            return max(1, len(text) // 4)
        
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error encoding text: {e}, falling back to estimation")
            return max(1, len(text) // 4)
    
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """
        Count tokens in messages using OpenAI's message format.
        Based on OpenAI's cookbook examples.
        """
        if self._encoding is None:
            # Fallback estimation
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            return max(1, total_chars // 4)
        
        try:
            # Tokens per message overhead varies by model
            if "gpt-3.5-turbo" in self.model_name:
                tokens_per_message = 4
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-4" in self.model_name:
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                tokens_per_message = 3
                tokens_per_name = 1
            
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if isinstance(value, str):
                        num_tokens += len(self._encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens
            
        except Exception as e:
            logger.warning(f"Error counting message tokens: {e}")
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            return max(1, total_chars // 4)
    
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        if max_tokens:
            return min(max_tokens, 4096)  # Cap at reasonable default
        return 150  # Conservative default


class AnthropicTokenCalculator(TokenCalculator):
    """Token calculator for Anthropic models"""
    
    def __init__(self, model_name: str = "claude-3-sonnet"):
        self.model_name = model_name
        self._client = None
        self._load_client()
    
    def _load_client(self):
        """Load Anthropic client for token counting"""
        try:
            import anthropic
            import os
            # Initialize client with API key from environment or config
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = None
                logger.warning("Anthropic API key not found for token counting")
        except ImportError:
            logger.warning("Anthropic client not available, using estimation")
            self._client = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's method or estimation"""
        if self._client:
            try:
                # Use Anthropic's messages.count_tokens method
                response = self._client.messages.count_tokens(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}]
                )
                return response.input_tokens
            except Exception as e:
                logger.debug(f"Error using Anthropic token counting: {e}, falling back to estimation")
        
        # Fallback: Anthropic uses roughly 3.5 characters per token
        return max(1, len(text) * 3 // 10)
    
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages"""
        if self._client:
            try:
                # Convert to Anthropic format
                anthropic_messages = []
                system_message = None
                
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    if role == 'system':
                        system_message = content
                    else:
                        # Anthropic uses 'user' and 'assistant' roles
                        anthropic_role = 'assistant' if role == 'assistant' else 'user'
                        anthropic_messages.append({
                            "role": anthropic_role,
                            "content": content
                        })
                
                # Use Anthropic's messages.count_tokens method
                kwargs = {
                    "model": self.model_name,
                    "messages": anthropic_messages if anthropic_messages else [{"role": "user", "content": ""}]
                }
                if system_message:
                    kwargs["system"] = system_message
                    
                response = self._client.messages.count_tokens(**kwargs)
                return response.input_tokens
                
            except Exception as e:
                logger.debug(f"Error counting Anthropic message tokens: {e}, falling back to estimation")
        
        # Fallback estimation
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        return max(1, total_chars * 3 // 10)
    
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        if max_tokens:
            return min(max_tokens, 4096)
        return 200  # Conservative default for Claude


class GoogleTokenCalculator(TokenCalculator):
    """Token calculator for Google models (Gemini)"""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Google's estimation"""
        try:
            import google.generativeai as genai
            
            # Use Google's token counting if available
            model = genai.GenerativeModel(self.model_name)
            result = model.count_tokens(text)
            return result.total_tokens
            
        except Exception as e:
            logger.warning(f"Error using Google token counting: {e}")
            # Google uses roughly 4 characters per token
            return max(1, len(text) // 4)
    
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages"""
        try:
            import google.generativeai as genai
            
            # Convert messages to Google format
            conversation_text = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_text += f"{role}: {content}\n"
            
            model = genai.GenerativeModel(self.model_name)
            result = model.count_tokens(conversation_text)
            return result.total_tokens
            
        except Exception as e:
            logger.warning(f"Error counting Google message tokens: {e}")
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            return max(1, total_chars // 4)
    
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        if max_tokens:
            return min(max_tokens, 2048)
        return 150


class CohereTokenCalculator(TokenCalculator):
    """Token calculator for Cohere models"""
    
    def __init__(self, model_name: str = "command"):
        self.model_name = model_name
        self._client = None
        self._load_client()
    
    def _load_client(self):
        """Load Cohere client"""
        try:
            import cohere
            self._client = cohere.Client()
        except ImportError:
            logger.warning("Cohere client not available")
            self._client = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Cohere's tokenizer"""
        if self._client:
            try:
                response = self._client.tokenize(text, model=self.model_name)
                return len(response.tokens)
            except Exception as e:
                logger.warning(f"Error using Cohere tokenizer: {e}")
        
        # Fallback estimation (Cohere uses ~4.5 chars per token)
        return max(1, len(text) * 2 // 9)
    
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages"""
        conversation_text = ""
        for msg in messages:
            content = msg.get('content', '')
            conversation_text += content + "\n"
        
        return self.count_tokens(conversation_text)
    
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        if max_tokens:
            return min(max_tokens, 2048)
        return 150


class FallbackTokenCalculator(TokenCalculator):
    """Fallback calculator using simple heuristics"""
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def count_tokens(self, text: str) -> int:
        """Simple character-based estimation"""
        return max(1, int(len(text) / self.chars_per_token))
    
    def count_message_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in messages using simple estimation"""
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        # Add overhead for message formatting
        overhead = len(messages) * 10
        return max(1, int((total_chars + overhead) / self.chars_per_token))
    
    def estimate_completion_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate completion tokens"""
        if max_tokens:
            return max_tokens
        return 100


class TokenCalculatorFactory:
    """Factory for creating appropriate token calculators"""
    
    @staticmethod
    def create_calculator(
        provider: Union[str, ProviderType],
        model_name: str
    ) -> TokenCalculator:
        """Create a token calculator for the given provider and model"""
        
        if isinstance(provider, str):
            try:
                provider = ProviderType(provider.lower())
            except ValueError:
                logger.warning(f"Unknown provider {provider}, using fallback calculator")
                return FallbackTokenCalculator()
        
        try:
            if provider == ProviderType.OPENAI:
                return OpenAITokenCalculator(model_name)
            elif provider == ProviderType.ANTHROPIC:
                return AnthropicTokenCalculator(model_name)
            elif provider == ProviderType.GOOGLE:
                return GoogleTokenCalculator(model_name)
            elif provider == ProviderType.COHERE:
                return CohereTokenCalculator(model_name)
            else:
                logger.warning(f"No specific calculator for {provider}, using fallback")
                return FallbackTokenCalculator()
                
        except Exception as e:
            logger.warning(f"Error creating calculator for {provider}: {e}, using fallback")
            return FallbackTokenCalculator()


# Enhanced rate limiter integration
import time
from typing import Optional
from .base import Middleware
from ..core.types import CompletionRequest, CompletionResponse
from ..core.exceptions import LLMRateLimitError

class RateLimitMiddleware(Middleware):
    """
    Enhanced rate limit middleware using actual token calculators
    """
    
    def __init__(self, rate_limiter, provider: str, model_name: str):
        self.rate_limiter = rate_limiter
        self.provider = provider
        self.model_name = model_name
        self.token_calculator = TokenCalculatorFactory.create_calculator(
            provider, model_name
        )
        logger.info(f"Using {type(self.token_calculator).__name__} for {provider}/{model_name}")
    
    async def process_request(self, request: CompletionRequest, context: dict) -> CompletionRequest:
        """Enhanced request processing with accurate token counting"""
        
        # Use actual tokenizer for accurate estimation
        estimated_input = 0
        if hasattr(request, 'messages') and request.messages:
            estimated_input = self.token_calculator.count_message_tokens(
                [{"role": msg.role, "content": msg.content} for msg in request.messages]
            )
        elif hasattr(request, 'prompt') and request.prompt:
            estimated_input = self.token_calculator.count_tokens(request.prompt)
        
        estimated_output = self.token_calculator.estimate_completion_tokens(
            getattr(request, 'max_tokens', None)
        )
        
        # Store for comparison with actual usage
        context['estimated_input_tokens'] = estimated_input
        context['estimated_output_tokens'] = estimated_output
        context['request_timestamp'] = time.time()
        
        logger.debug(
            f"Token estimates - Input: {estimated_input}, Output: {estimated_output} "
            f"(using {type(self.token_calculator).__name__})"
        )
        
        # Acquire rate limit with accurate estimates
        acquired = await self.rate_limiter.acquire_for_request(
            estimated_input=estimated_input,
            estimated_output=estimated_output,
            timeout=30.0
        )
        
        if not acquired:
            from ..core.exceptions import LLMRateLimitError
            raise LLMRateLimitError("Rate limit exceeded")
        
        return request
    
    async def process_response(self, response: CompletionResponse, context: dict) -> CompletionResponse:
        """Process response with actual usage reporting"""
        if hasattr(response, 'usage') and response.usage:
            # Use actual token counts from provider
            input_tokens = response.usage.get('prompt_tokens', 0)
            output_tokens = response.usage.get('completion_tokens', 0)
            
            await self.rate_limiter.report_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Log accuracy of estimates
            estimated_input = context.get('estimated_input_tokens', 0)
            estimated_output = context.get('estimated_output_tokens', 0)
            
            input_accuracy = (estimated_input / max(1, input_tokens)) if input_tokens > 0 else 1.0
            output_accuracy = (estimated_output / max(1, output_tokens)) if output_tokens > 0 else 1.0
            
            logger.debug(
                f"Token usage - Actual: {input_tokens}/{output_tokens}, "
                f"Estimated: {estimated_input}/{estimated_output}, "
                f"Accuracy: {input_accuracy:.2f}/{output_accuracy:.2f}"
            )
        
        return response
    
    async def process_error(self, error: Exception, context: dict) -> Optional[Exception]:
        """Handle rate limit errors"""
        if isinstance(error, LLMRateLimitError):
            logger.warning(f"Rate limit error for {self.provider}/{self.model_name}: {error}")
        return error


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_calculators():
        """Test different token calculators"""
        
        test_text = "Hello, how are you doing today? I hope you're having a great day!"
        test_messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "Thank you for the information!"}
        ]
        
        providers = ["openai", "anthropic", "google", "cohere"]
        models = ["gpt-4", "claude-3-sonnet", "gemini-pro", "command"]
        
        for provider, model in zip(providers, models):
            print(f"\n--- {provider.upper()} ({model}) ---")
            
            calculator = TokenCalculatorFactory.create_calculator(provider, model)
            
            text_tokens = calculator.count_tokens(test_text)
            message_tokens = calculator.count_message_tokens(test_messages)
            completion_estimate = calculator.estimate_completion_tokens(100)
            
            print(f"Text tokens: {text_tokens}")
            print(f"Message tokens: {message_tokens}")
            print(f"Completion estimate: {completion_estimate}")
            print(f"Calculator type: {type(calculator).__name__}")
    
    # Run the test
    asyncio.run(test_calculators())