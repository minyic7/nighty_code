# src/llm/providers/anthropic.py
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


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider implementation"""

    async def _initialize_provider(self):
        """Initialize the Anthropic client"""
        try:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

            # Mark as available without validation (validation can fail due to model name issues)
            self._status.is_available = True
            logger.info(f"Anthropic client {self._client_id} initialized successfully")

        except ImportError:
            raise LLMProviderError(
                "Anthropic library not installed. Run: pip install anthropic"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise LLMProviderError(f"Failed to initialize Anthropic client: {e}")

    async def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Anthropic Claude"""
        try:
            logger.debug(f"Client {self._client_id} starting completion request")
            
            # Convert messages to Anthropic format
            system_message = None
            messages = []

            for msg in request.messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                else:
                    messages.append({
                        "role": msg.role.value if msg.role != MessageRole.USER else "user",
                        "content": msg.content
                    })

            # Merge configuration
            temperature = request.temperature or self.config.temperature
            max_tokens = request.max_tokens or self.config.max_tokens or 1024

            # Prepare API call parameters
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **request.extra_params,
                **self.config.extra_params
            }

            if system_message:
                params["system"] = system_message

            if request.stop:
                params["stop_sequences"] = request.stop

            # Make the API call
            logger.info(f"Client {self._client_id} making API call to Anthropic")
            response = await self.client.messages.create(**params)
            logger.info(f"Client {self._client_id} received response from Anthropic")

            # Update status
            self._status.total_requests += 1
            self._status.last_used = datetime.now()

            # Extract content from response
            content = ""
            if response.content:
                # Handle different content types
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text
                    elif isinstance(block, str):
                        content += block

            
            # Build response
            return CompletionResponse(
                content=content,
                model=response.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                } if hasattr(response, 'usage') else {},
                metadata={
                    "stop_reason": response.stop_reason if hasattr(response, 'stop_reason') else None,
                    "id": response.id,
                },
                client_id=self._client_id  # Include client ID in response
            )

        except Exception as e:
            self._status.error_count += 1
            logger.error(f"Anthropic completion error: {e}")

            # Handle specific error types
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
            elif "rate limit" in str(e).lower():
                raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Anthropic request timed out: {e}")
            else:
                raise LLMProviderError(f"Anthropic completion failed: {e}")

    async def _do_stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming completion using Anthropic Claude"""
        try:
            # Convert messages to Anthropic format
            system_message = None
            messages = []

            for msg in request.messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                else:
                    messages.append({
                        "role": msg.role.value if msg.role != MessageRole.USER else "user",
                        "content": msg.content
                    })

            # Merge configuration
            temperature = request.temperature or self.config.temperature
            max_tokens = request.max_tokens or self.config.max_tokens or 1024

            # Prepare API call parameters
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **request.extra_params,
                **self.config.extra_params
            }

            if system_message:
                params["system"] = system_message

            if request.stop:
                params["stop_sequences"] = request.stop

            # Make the streaming API call
            stream = await self.client.messages.create(**params, stream=True)

            # Update status
            self._status.total_requests += 1
            self._status.last_used = datetime.now()

            # Stream the response
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield event.delta.text

        except Exception as e:
            self._status.error_count += 1
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMProviderError(f"Anthropic streaming failed: {e}")

    async def validate_connection(self) -> bool:
        """Validate the connection to Anthropic"""
        try:
            # Try a minimal API call to validate the connection
            # Anthropic doesn't have a simple model list endpoint like OpenAI
            # We'll do a minimal completion request
            response = await self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )
            return bool(response.id)
        except Exception as e:
            logger.warning(f"Anthropic connection validation failed: {e}")
            return False

    async def _close_provider(self):
        """Close the Anthropic client"""
        try:
            if self.client:
                # Anthropic client doesn't have an explicit close method
                self.client = None
                self._status.is_available = False
                logger.info(f"Anthropic client {self._client_id} closed")
        except Exception as e:
            logger.error(f"Error closing Anthropic client: {e}")