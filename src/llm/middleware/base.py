"""
Base middleware interface for LLM request/response processing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Any
import asyncio
import logging

from ..core.types import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base class for middleware components."""
    
    @abstractmethod
    async def process_request(
        self,
        request: CompletionRequest,
        context: dict
    ) -> CompletionRequest:
        """
        Process request before it reaches the provider.
        
        Args:
            request: The completion request
            context: Shared context for the request lifecycle
            
        Returns:
            Modified request or original request
        """
        pass
    
    @abstractmethod
    async def process_response(
        self,
        response: CompletionResponse,
        context: dict
    ) -> CompletionResponse:
        """
        Process response after it comes from the provider.
        
        Args:
            response: The completion response
            context: Shared context for the request lifecycle
            
        Returns:
            Modified response or original response
        """
        pass
    
    @abstractmethod
    async def process_error(
        self,
        error: Exception,
        context: dict
    ) -> Optional[Exception]:
        """
        Process errors that occur during request processing.
        
        Args:
            error: The exception that occurred
            context: Shared context for the request lifecycle
            
        Returns:
            Modified error, original error, or None to suppress
        """
        pass


class MiddlewareChain:
    """
    Chain of middleware components that process requests/responses in order.
    """
    
    def __init__(self, middlewares: Optional[List[Middleware]] = None):
        self.middlewares = middlewares or []
        self._logger = logger
    
    def add(self, middleware: Middleware):
        """Add middleware to the chain."""
        self.middlewares.append(middleware)
        return self
    
    def remove(self, middleware_type: type):
        """Remove middleware of a specific type."""
        self.middlewares = [
            m for m in self.middlewares 
            if not isinstance(m, middleware_type)
        ]
        return self
    
    async def execute_request(
        self,
        request: CompletionRequest,
        handler: Callable,
        context: Optional[dict] = None
    ) -> CompletionResponse:
        """
        Execute the middleware chain for a request.
        
        Args:
            request: The completion request
            handler: The actual completion handler
            context: Optional shared context
            
        Returns:
            The completion response
        """
        context = context or {}
        
        # Process request through middleware chain (forward)
        processed_request = request
        for middleware in self.middlewares:
            try:
                processed_request = await middleware.process_request(
                    processed_request, context
                )
            except Exception as e:
                self._logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")
                # Let error middleware handle it
                error = await self._process_error(e, context)
                if error:
                    raise error
                # If error was suppressed, continue with original request
                break
        
        try:
            # Execute the actual request
            response = await handler(processed_request)
            
            # Process response through middleware chain (reverse)
            processed_response = response
            for middleware in reversed(self.middlewares):
                try:
                    processed_response = await middleware.process_response(
                        processed_response, context
                    )
                except Exception as e:
                    self._logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")
                    # Continue with original response if middleware fails
                    continue
            
            return processed_response
            
        except Exception as e:
            # Process error through middleware chain
            error = await self._process_error(e, context)
            if error:
                raise error
            # If all middleware suppressed the error, raise original
            raise e
    
    async def _process_error(
        self,
        error: Exception,
        context: dict
    ) -> Optional[Exception]:
        """Process error through middleware chain."""
        current_error = error
        
        for middleware in reversed(self.middlewares):
            try:
                result = await middleware.process_error(current_error, context)
                if result is None:
                    # Error was suppressed
                    return None
                current_error = result
            except Exception as e:
                self._logger.error(f"Error in error handler {middleware.__class__.__name__}: {e}")
                continue
        
        return current_error


class NoOpMiddleware(Middleware):
    """Middleware that does nothing (for testing)."""
    
    async def process_request(self, request: CompletionRequest, context: dict) -> CompletionRequest:
        return request
    
    async def process_response(self, response: CompletionResponse, context: dict) -> CompletionResponse:
        return response
    
    async def process_error(self, error: Exception, context: dict) -> Optional[Exception]:
        return error