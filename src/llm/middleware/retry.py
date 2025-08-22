"""
Retry middleware for handling transient failures.
"""

import asyncio
import random
from typing import Optional, Set
import logging

from .base import Middleware
from ..core.types import CompletionRequest, CompletionResponse
from ..core.exceptions import (
    LLMException,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError
)

logger = logging.getLogger(__name__)


class RetryMiddleware(Middleware):
    """
    Middleware that implements retry logic with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Set[type]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        # Default retryable exceptions
        self.retryable_exceptions = retryable_exceptions or {
            LLMTimeoutError,
            LLMConnectionError,
            LLMRateLimitError,
        }
    
    async def process_request(
        self,
        request: CompletionRequest,
        context: dict
    ) -> CompletionRequest:
        """Initialize retry context."""
        context['retry_count'] = context.get('retry_count', 0)
        context['total_retries'] = context.get('total_retries', 0)
        return request
    
    async def process_response(
        self,
        response: CompletionResponse,
        context: dict
    ) -> CompletionResponse:
        """Reset retry count on successful response."""
        if 'retry_count' in context:
            if context['retry_count'] > 0:
                logger.info(f"Request succeeded after {context['retry_count']} retries")
        return response
    
    async def process_error(
        self,
        error: Exception,
        context: dict
    ) -> Optional[Exception]:
        """Handle retryable errors with exponential backoff."""
        # Check if error is retryable
        if not self._is_retryable(error):
            return error
        
        retry_count = context.get('retry_count', 0)
        
        # Check if we've exceeded max retries
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded for error: {error}")
            return error
        
        # Calculate delay with exponential backoff
        delay = self._calculate_delay(retry_count)
        
        logger.warning(
            f"Retryable error occurred: {error}. "
            f"Retrying in {delay:.2f}s (attempt {retry_count + 1}/{self.max_retries})"
        )
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # Update context
        context['retry_count'] = retry_count + 1
        context['total_retries'] = context.get('total_retries', 0) + 1
        
        # Return None to signal retry
        return None
    
    def _is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        return any(isinstance(error, exc_type) for exc_type in self.retryable_exceptions)
    
    def _calculate_delay(self, retry_count: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** retry_count),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (±25% of delay)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)  # Ensure non-negative