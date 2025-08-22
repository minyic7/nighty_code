# src/llm/core/client.py
from typing import Optional, List, AsyncIterator, Dict, Any
from contextlib import asynccontextmanager
import logging

from .types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    LLMProvider
)
from .pool import LLMConnectionPool
from .config import LLMConfig
from .exceptions import LLMException
from ..utils import AsyncContextManagerMixin


logger = logging.getLogger(__name__)


class LLMClient(AsyncContextManagerMixin):
    """High-level client interface for LLM operations"""
    
    def __init__(self, pool: LLMConnectionPool):
        self.pool = pool
        self.provider = pool.config.provider
        self.model = pool.config.model
    
    async def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion"""
        logger.debug(f"LLMClient.complete called with {len(messages)} messages")
        
        request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stop=stop,
            extra_params=kwargs
        )
        
        logger.debug(f"Getting client from pool...")
        async with self.pool.get_client() as client:
            logger.debug(f"Got client {client._client_id}, calling complete...")
            response = await client.complete(request)
            logger.debug(f"Client {client._client_id} completed request")
            return response
    
    async def stream_complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming completion"""
        request = CompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stop=stop,
            extra_params=kwargs
        )
        
        async with self.pool.get_client() as client:
            async for chunk in client.stream_complete(request):
                yield chunk
    
    async def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        history: Optional[List[Message]] = None,
        **kwargs
    ) -> str:
        """Simple chat interface"""
        messages = []
        
        if system_message:
            messages.append(Message(MessageRole.SYSTEM, system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(Message(MessageRole.USER, user_message))
        
        response = await self.complete(messages, **kwargs)
        return response.content
    
    async def stream_chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        history: Optional[List[Message]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Simple streaming chat interface"""
        messages = []
        
        if system_message:
            messages.append(Message(MessageRole.SYSTEM, system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(Message(MessageRole.USER, user_message))
        
        async for chunk in self.stream_complete(messages, **kwargs):
            yield chunk
    
    async def close(self):
        """Close the client and its pool"""
        await self.pool.close()
    
    # Status tracking moved to pool and manager levels
    # Removed create() method - clients should be created through LLMManager
    # Context manager methods inherited from AsyncContextManagerMixin