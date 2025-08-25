# src/llm/providers/openai.py
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation"""
    
    async def _initialize_provider(self):
        """Initialize the OpenAI client"""
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Validate the connection
            if await self.validate_connection():
                self._status.is_available = True
                logger.info(f"OpenAI client {self._client_id} initialized successfully")
            else:
                raise LLMProviderError("Failed to validate OpenAI connection")
                
        except ImportError:
            raise LLMProviderError(
                "OpenAI library not installed. Run: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise LLMProviderError(f"Failed to initialize OpenAI client: {e}")
    
    async def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using OpenAI"""
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
            
            
            # Build response
            return CompletionResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=LLMProvider.OPENAI,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else {},
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "id": response.id,
                }
            )
            
        except Exception as e:
            self._status.error_count += 1
            logger.error(f"OpenAI completion error: {e}")
            
            # Handle specific error types
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
            elif "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(f"OpenAI request timed out: {e}")
            else:
                raise LLMProviderError(f"OpenAI completion failed: {e}")
    
    async def _do_stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming completion using OpenAI"""
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
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._status.error_count += 1
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}")
    
    async def validate_connection(self) -> bool:
        """Validate the connection to OpenAI - Fixed to use models.list() as fallback"""
        try:
            # Method 1: Try the original validation first (works for standard OpenAI)
            try:
                response = await self.client.models.retrieve(self.config.model)
                if response.id == self.config.model:
                    return True
            except Exception:
                pass  # Fallback to Method 2
            
            # Method 2: List models and check if our model exists (works for proxies/Azure)
            models_response = await self.client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if self.config.model in available_models:
                logger.info(f"Model '{self.config.model}' validated using models.list()")
                return True
            else:
                logger.warning(f"Model '{self.config.model}' not found in available models")
                return False
                
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            return False
    
    async def _close_provider(self):
        """Close the OpenAI client"""
        try:
            if self.client:
                await self.client.close()
                self._status.is_available = False
                logger.info(f"OpenAI client {self._client_id} closed")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")