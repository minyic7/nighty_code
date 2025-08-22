# src/llm/providers/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Any, Dict
import asyncio
from datetime import datetime
import logging

from ..core.types import (
    LLMConfig,
    CompletionRequest,
    CompletionResponse,
    ClientStatus
)
from ..core.exceptions import LLMException, LLMRateLimitError
from ..middleware.rate_limiter import ClientRateLimiter, RateLimitConfig
from ..middleware.token_calculator import TokenCalculatorFactory
from ..middleware.base import MiddlewareChain
from ..middleware.logging import LoggingMiddleware
from ..middleware.metrics import MetricsMiddleware
from ..middleware.retry import RetryMiddleware

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers with middleware support"""

    def __init__(self, config: LLMConfig, rate_limit_config: Optional[RateLimitConfig] = None):
        self.config = config
        self.client = None
        self._client_id = self._generate_client_id()
        self._status = ClientStatus(
            client_id=self._client_id,
            provider=config.provider,
            model=config.model,
            is_available=False,
            in_use=False
        )
        self._lock = asyncio.Lock()
        
        # Initialize middleware chain
        self.middleware_chain = MiddlewareChain()
        self.rate_limiter: Optional[ClientRateLimiter] = None
        
        # Add rate limiter middleware if config provided
        if rate_limit_config:
            self.rate_limiter = ClientRateLimiter(self._client_id, rate_limit_config)
            # Create the enhanced rate limit middleware with token calculator
            from ..middleware.token_calculator import RateLimitMiddleware
            # Set API key in environment for token calculator
            import os
            os.environ['ANTHROPIC_API_KEY'] = config.api_key
            rate_limit_middleware = RateLimitMiddleware(
                self.rate_limiter,
                config.provider.value,
                config.model
            )
            self.middleware_chain.add(rate_limit_middleware)
            logger.debug(f"Enhanced rate limiting enabled for {self._client_id} with accurate token counting")

    def _generate_client_id(self) -> str:
        """Generate a unique client ID"""
        import uuid
        return f"{self.config.provider.value}_{uuid.uuid4().hex[:8]}"

    async def initialize(self):
        """Initialize the provider client"""
        # Start rate limiter if present
        if self.rate_limiter:
            await self.rate_limiter.start()
            logger.info(f"Rate limiter started for {self._client_id}")
        
        # Call provider-specific initialization
        await self._initialize_provider()
    
    @abstractmethod
    async def _initialize_provider(self):
        """Provider-specific initialization"""
        pass

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion with middleware processing"""
        logger.debug(f"Provider {self._client_id} complete() called, middleware count: {len(self.middleware_chain.middlewares)}")
        
        # Define the actual handler
        async def handler(processed_request: CompletionRequest) -> CompletionResponse:
            logger.debug(f"Provider {self._client_id} handler calling _do_complete")
            return await self._do_complete(processed_request)
        
        # Execute through middleware chain
        if self.middleware_chain.middlewares:
            logger.debug(f"Provider {self._client_id} executing through middleware chain")
            return await self.middleware_chain.execute_request(
                request=request,
                handler=handler,
                context={'client_id': self._client_id}
            )
        else:
            # No middleware, call directly
            logger.debug(f"Provider {self._client_id} no middleware, calling _do_complete directly")
            return await self._do_complete(request)

    @abstractmethod
    async def _do_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Provider-specific completion implementation"""
        pass

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion with middleware processing"""
        # For streaming, process request through middleware first
        context = {'client_id': self._client_id}
        processed_request = request
        
        # Process request through middleware
        for middleware in self.middleware_chain.middlewares:
            try:
                processed_request = await middleware.process_request(processed_request, context)
            except Exception as e:
                logger.error(f"Middleware error in streaming: {e}")
                raise
        
        # Do the actual streaming
        async for chunk in self._do_stream_complete(processed_request):
            yield chunk
    
    @abstractmethod
    async def _do_stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Provider-specific streaming implementation"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the provider"""
        pass

    async def close(self):
        """Close the provider client"""
        # Stop rate limiter if present
        if self.rate_limiter:
            await self.rate_limiter.stop()
            logger.debug(f"Rate limiter stopped for {self._client_id}")
        
        # Call provider-specific close
        await self._close_provider()
    
    @abstractmethod
    async def _close_provider(self):
        """Provider-specific close implementation"""
        pass

    async def acquire(self) -> 'BaseLLMProvider':
        """Acquire this client for use"""
        async with self._lock:
            if self._status.in_use:
                raise LLMException(f"Client {self._client_id} is already in use")
            self._status.in_use = True
            self._status.last_used = datetime.now()
            logger.debug(f"Client {self._client_id} acquired")
        return self

    async def release(self):
        """Release this client after use"""
        async with self._lock:
            self._status.in_use = False
            logger.debug(f"Client {self._client_id} released")

    def get_status(self) -> ClientStatus:
        """Get the current status of this client"""
        return self._status

    async def health_check(self) -> bool:
        """Perform a health check on this client"""
        try:
            result = await self.validate_connection()
            self._status.is_available = result
            if result:
                self._status.error_count = 0
            return result
        except Exception as e:
            logger.error(f"Health check failed for {self._client_id}: {e}")
            self._status.is_available = False
            self._status.error_count += 1
            return False

    def __enter__(self):
        raise TypeError("Use async with instead")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
    
    def get_rate_limit_status(self) -> Optional[Dict]:
        """Get current rate limit status"""
        if self.rate_limiter:
            return self.rate_limiter.get_metrics()
        return None