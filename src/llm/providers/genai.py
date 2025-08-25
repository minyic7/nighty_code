# src/llm/providers/genai.py
from typing import AsyncIterator, Optional, Dict, Any
import logging
from datetime import datetime

from .base import BaseLLMProvider
from ..core.types import (
    LLMConfig,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    MessageRole
)
from ..core.exceptions import (
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError
)


logger = logging.getLogger(__name__)


class GenAIProvider(BaseLLMProvider):
    """GenAI LLM provider implementation using OpenAI-compatible API"""
    
    async def _initialize_provider(self):
        """Initialize the GenAI client using OpenAI SDK with custom base URL"""
        try:
            from openai import AsyncOpenAI
            
            # GenAI uses OpenAI-compatible API with custom base URL
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Skip validation for GenAI as it might not support model listing
            # The connection will be validated on first request
            self._status.is_available = True
            logger.info(f"GenAI client {self._client_id} initialized with base_url: {self.config.base_url}")
                
        except ImportError:
            raise LLMProviderError(
                "OpenAI library not installed. Run: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize GenAI client: {e}")
            raise LLMProviderError(f"Failed to initialize GenAI client: {e}")
    
    async def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using GenAI"""
        try:
            # Prepare messages
            messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.messages
            ]
            
            # Merge configuration
            temperature = request.temperature or self.config.temperature
            max_tokens = request.max_tokens or self.config.max_tokens
            
            # Prepare API call parameters
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                **request.extra_params,
                **self.config.extra_params
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            if request.stop:
                params["stop"] = request.stop
            
            # Make the API call
            response = await self.client.chat.completions.create(**params)
            
            # Update status
            self._status.total_requests += 1
            self._status.last_used = datetime.now()
            
            # Build response - handle GenAI's response format
            usage_data = {}
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
            
            return CompletionResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=LLMProvider.GENAI,
                usage=usage_data,
                metadata={
                    "finish_reason": response.choices[0].finish_reason if response.choices else None,
                    "id": getattr(response, 'id', None),
                },
                client_id=self._client_id
            )
            
        except Exception as e:
            self._status.error_count += 1
            logger.error(f"GenAI completion error: {e}")
            
            # Handle specific error types
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str:
                raise LLMAuthenticationError(f"GenAI authentication failed: {e}")
            elif "rate limit" in error_str:
                raise LLMRateLimitError(f"GenAI rate limit exceeded: {e}")
            elif "timeout" in error_str:
                raise LLMTimeoutError(f"GenAI request timed out: {e}")
            else:
                raise LLMProviderError(f"GenAI completion failed: {e}")
    
    async def _do_stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming completion using GenAI"""
        try:
            # Prepare messages
            messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.messages
            ]
            
            # Merge configuration
            temperature = request.temperature or self.config.temperature
            max_tokens = request.max_tokens or self.config.max_tokens
            
            # Prepare API call parameters
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                **request.extra_params,
                **self.config.extra_params
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            if request.stop:
                params["stop"] = request.stop
            
            # Make the streaming API call
            stream = await self.client.chat.completions.create(**params)
            
            # Update status
            self._status.total_requests += 1
            self._status.last_used = datetime.now()
            
            # Stream the response
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
                    
        except Exception as e:
            self._status.error_count += 1
            logger.error(f"GenAI streaming error: {e}")
            raise LLMProviderError(f"GenAI streaming failed: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate the connection to GenAI"""
        try:
            # For GenAI, we'll do a simple test completion instead of model listing
            # as the proxy might not support the models endpoint
            test_messages = [
                {"role": "user", "content": "Test"}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=test_messages,
                max_tokens=5
            )
            
            if response and response.choices:
                logger.info(f"GenAI connection validated for model '{self.config.model}'")
                return True
            return False
                
        except Exception as e:
            logger.warning(f"GenAI connection validation failed: {e}")
            # Don't fail initialization just because validation failed
            # The actual request might still work
            return True  # Return True to allow initialization to continue
    
    async def _close_provider(self):
        """Close the GenAI client"""
        try:
            if self.client:
                await self.client.close()
                self._status.is_available = False
                logger.info(f"GenAI client {self._client_id} closed")
        except Exception as e:
            logger.error(f"Error closing GenAI client: {e}")