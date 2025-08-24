# src/llm/core/client.py
from typing import Optional, List, AsyncIterator, Dict, Any, Union, Type, TypeVar, overload
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel

from .types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    LLMProvider
)
from .pool import LLMConnectionPool
from .config import LLMConfig, InstructorConfig
from .exceptions import LLMException
from ..utils import AsyncContextManagerMixin


logger = logging.getLogger(__name__)

# Type variable for generic Pydantic models
T = TypeVar('T', bound=BaseModel)


class LLMClient(AsyncContextManagerMixin):
    """High-level client interface for LLM operations with optional Instructor support"""
    
    def __init__(self, pool: LLMConnectionPool, instructor_config: Optional[InstructorConfig] = None):
        self.pool = pool
        self.provider = pool.config.provider
        self.model = pool.config.model
        self.instructor_config = instructor_config or InstructorConfig()
        self._instructor = None  # Lazy-loaded
        self._instructor_available = None  # Cache availability check
    
    @property
    def instructor(self):
        """Lazy-load Instructor when needed"""
        if self._instructor is None and self._check_instructor_available():
            try:
                import instructor
                # We'll patch the client when making the actual call
                self._instructor = instructor
                logger.info("Instructor loaded successfully")
            except ImportError:
                logger.warning("Instructor not available, falling back to normal completion")
                self._instructor_available = False
        return self._instructor
    
    def _check_instructor_available(self) -> bool:
        """Check if Instructor is available and enabled"""
        if self._instructor_available is not None:
            return self._instructor_available
            
        if not self.instructor_config.enabled:
            self._instructor_available = False
            return False
            
        try:
            import instructor
            self._instructor_available = True
            return True
        except ImportError:
            self._instructor_available = False
            return False
    
    @overload
    async def complete(
        self,
        messages: List[Message],
        *,
        response_model: None = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResponse: ...
    
    @overload
    async def complete(
        self,
        messages: List[Message],
        *,
        response_model: Type[T],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> T: ...
    
    async def complete(
        self,
        messages: List[Message],
        *,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[CompletionResponse, BaseModel]:
        """
        Generate a completion with optional structured output.
        
        Args:
            messages: List of messages
            response_model: Optional Pydantic model for structured output (triggers Instructor)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            CompletionResponse if no response_model, instance of response_model otherwise
        """
        # Auto-detect Instructor usage based on response_model
        if response_model is not None and self._check_instructor_available():
            logger.debug(f"Using Instructor for structured output: {response_model.__name__}")
            return await self._complete_with_instructor(
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
        elif response_model is not None and not self._check_instructor_available():
            logger.warning(
                f"response_model provided ({response_model.__name__}) but Instructor not available. "
                "Install with: pip install instructor"
            )
            # Fall through to normal completion
        
        # Normal completion
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
    
    async def _complete_with_instructor(
        self,
        messages: List[Message],
        response_model: Type[BaseModel],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> BaseModel:
        """Complete with Instructor for structured output"""
        import instructor
        
        # Get a client from the pool
        async with self.pool.get_client() as pool_client:
            # Get the underlying provider client
            # The pool_client is the provider instance, which has a 'client' attribute
            provider_client = pool_client.client
            
            # Patch the client with Instructor
            if self.provider == LLMProvider.ANTHROPIC:
                # Anthropic-specific patching
                patched_client = instructor.from_anthropic(provider_client)
            elif self.provider == LLMProvider.OPENAI:
                # OpenAI-specific patching
                patched_client = instructor.from_openai(provider_client)
            else:
                # Generic patching
                patched_client = instructor.patch(provider_client)
            
            # Convert our Message objects to provider format
            provider_messages = self._convert_messages_for_provider(messages)
            
            try:
                # Use Instructor to get structured output
                if self.provider == LLMProvider.ANTHROPIC:
                    result = await patched_client.messages.create(
                        model=self.model,
                        messages=provider_messages,
                        response_model=response_model,
                        max_tokens=max_tokens or 1024,
                        temperature=temperature or 0.7,  # Default temperature if None
                        max_retries=self.instructor_config.max_retries,
                        **kwargs
                    )
                elif self.provider == LLMProvider.OPENAI:
                    result = await patched_client.chat.completions.create(
                        model=self.model,
                        messages=provider_messages,
                        response_model=response_model,
                        max_tokens=max_tokens,
                        temperature=temperature or 0.7,  # Default temperature if None
                        max_retries=self.instructor_config.max_retries,
                        **kwargs
                    )
                else:
                    # Generic call
                    result = await patched_client.create(
                        model=self.model,
                        messages=provider_messages,
                        response_model=response_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_retries=self.instructor_config.max_retries,
                        **kwargs
                    )
                
                logger.debug(f"Instructor returned structured result: {type(result).__name__}")
                return result
                
            except Exception as e:
                logger.error(f"Instructor failed: {e}")
                # Bubble up the error as requested
                raise LLMException(f"Instructor structured output failed: {e}") from e
    
    def _convert_messages_for_provider(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert our Message objects to provider format"""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    
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
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Simple chat interface with optional structured output.
        
        Returns:
            String content if no response_model, instance of response_model otherwise
        """
        messages = []
        
        if system_message:
            messages.append(Message(MessageRole.SYSTEM, system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(Message(MessageRole.USER, user_message))
        
        if response_model:
            # Return the structured object directly
            return await self.complete(messages, response_model=response_model, **kwargs)
        else:
            # Return just the content string
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