"""
Rate limiting for LLM API calls with token-based tracking.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 50
    input_tokens_per_minute: int = 50000
    output_tokens_per_minute: int = 10000
    total_tokens_per_minute: Optional[int] = None
    burst_multiplier: float = 1.5  # Allow burst up to 1.5x the limit
    
    def __post_init__(self):
        """Calculate total tokens if not specified."""
        if self.total_tokens_per_minute is None:
            self.total_tokens_per_minute = (
                self.input_tokens_per_minute + self.output_tokens_per_minute
            )


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    
    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: float) -> bool:
        """Try to consume tokens from the bucket."""
        self._refill()
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_time(self, tokens: float) -> float:
        """Calculate wait time for the requested tokens."""
        self._refill()
        if tokens <= self.tokens:
            return 0.0
        
        deficit = tokens - self.tokens
        return deficit / self.refill_rate
    
    def _refill(self):
        """Refill the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def available(self) -> float:
        """Get available tokens."""
        self._refill()
        return self.tokens


class ClientRateLimiter:
    """Rate limiter for a single client/API key."""
    
    def __init__(self, client_id: str, config: RateLimitConfig):
        self.client_id = client_id
        self.config = config
        
        # Create token buckets for different limits
        self.request_bucket = TokenBucket(
            capacity=config.requests_per_minute * config.burst_multiplier,
            refill_rate=config.requests_per_minute / 60.0
        )
        
        self.input_bucket = TokenBucket(
            capacity=config.input_tokens_per_minute * config.burst_multiplier,
            refill_rate=config.input_tokens_per_minute / 60.0
        )
        
        self.output_bucket = TokenBucket(
            capacity=config.output_tokens_per_minute * config.burst_multiplier,
            refill_rate=config.output_tokens_per_minute / 60.0
        )
        
        self.total_bucket = TokenBucket(
            capacity=config.total_tokens_per_minute * config.burst_multiplier,
            refill_rate=config.total_tokens_per_minute / 60.0
        )
        
        # Tracking
        self.request_history = deque(maxlen=100)
        self.token_history = deque(maxlen=100)
        self._lock = asyncio.Lock()
        self._started = False
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_wait_time': 0.0,
        }
    
    async def start(self):
        """Start the rate limiter."""
        self._started = True
        logger.debug(f"Rate limiter started for {self.client_id}")
    
    async def stop(self):
        """Stop the rate limiter."""
        self._started = False
        logger.debug(f"Rate limiter stopped for {self.client_id}")
    
    async def acquire_for_request(
        self,
        estimated_input: int,
        estimated_output: int,
        timeout: float = 30.0
    ) -> bool:
        """
        Try to acquire rate limit for a request.
        
        Args:
            estimated_input: Estimated input tokens
            estimated_output: Estimated output tokens
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        if not self._started:
            return True  # No rate limiting if not started
        
        start_time = time.time()
        total_tokens = estimated_input + estimated_output
        
        async with self._lock:
            while True:
                # Check all buckets
                can_proceed = (
                    self.request_bucket.consume(1) and
                    self.input_bucket.consume(estimated_input) and
                    self.output_bucket.consume(estimated_output) and
                    self.total_bucket.consume(total_tokens)
                )
                
                if can_proceed:
                    # Update metrics
                    self.metrics['total_requests'] += 1
                    self.metrics['accepted_requests'] += 1
                    
                    # Record request
                    self.request_history.append({
                        'timestamp': time.time(),
                        'estimated_input': estimated_input,
                        'estimated_output': estimated_output,
                    })
                    
                    wait_time = time.time() - start_time
                    self.metrics['total_wait_time'] += wait_time
                    
                    logger.debug(
                        f"Rate limit acquired for {self.client_id}: "
                        f"input={estimated_input}, output={estimated_output}, "
                        f"wait={wait_time:.2f}s"
                    )
                    return True
                
                # Calculate wait time
                wait_times = [
                    self.request_bucket.wait_time(1),
                    self.input_bucket.wait_time(estimated_input),
                    self.output_bucket.wait_time(estimated_output),
                    self.total_bucket.wait_time(total_tokens),
                ]
                max_wait = max(wait_times)
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed + max_wait > timeout:
                    # Restore tokens since we're not proceeding
                    self.request_bucket.tokens += 1
                    self.input_bucket.tokens += estimated_input
                    self.output_bucket.tokens += estimated_output
                    self.total_bucket.tokens += total_tokens
                    
                    self.metrics['rejected_requests'] += 1
                    logger.warning(
                        f"Rate limit timeout for {self.client_id}: "
                        f"would need {max_wait:.2f}s more"
                    )
                    return False
                
                # Wait for tokens to refill
                await asyncio.sleep(min(max_wait, 1.0))
    
    async def report_usage(
        self,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Report actual token usage after request completion.
        
        Args:
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens used
        """
        async with self._lock:
            self.metrics['total_input_tokens'] += input_tokens
            self.metrics['total_output_tokens'] += output_tokens
            
            # Record actual usage
            self.token_history.append({
                'timestamp': time.time(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            })
            
            logger.debug(
                f"Token usage reported for {self.client_id}: "
                f"input={input_tokens}, output={output_tokens}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and status."""
        return {
            'client_id': self.client_id,
            'config': {
                'requests_per_minute': self.config.requests_per_minute,
                'input_tokens_per_minute': self.config.input_tokens_per_minute,
                'output_tokens_per_minute': self.config.output_tokens_per_minute,
                'total_tokens_per_minute': self.config.total_tokens_per_minute,
            },
            'available': {
                'requests': self.request_bucket.available(),
                'input_tokens': self.input_bucket.available(),
                'output_tokens': self.output_bucket.available(),
                'total_tokens': self.total_bucket.available(),
            },
            'metrics': self.metrics.copy(),
            'recent_requests': len(self.request_history),
            'recent_tokens': len(self.token_history),
        }
    
    def reset_metrics(self):
        """Reset usage metrics."""
        self.metrics = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_wait_time': 0.0,
        }
        self.request_history.clear()
        self.token_history.clear()


class PoolRateLimiter:
    """Rate limiter for a pool of clients."""
    
    def __init__(self, configs: list[tuple[str, RateLimitConfig]]):
        """
        Initialize pool rate limiter.
        
        Args:
            configs: List of (client_id, rate_limit_config) tuples
        """
        self.limiters = {
            client_id: ClientRateLimiter(client_id, config)
            for client_id, config in configs
        }
        self._round_robin_index = 0
        self._client_ids = list(self.limiters.keys())
    
    async def start_all(self):
        """Start all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.start()
    
    async def stop_all(self):
        """Stop all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.stop()
    
    def get_next_limiter(self) -> ClientRateLimiter:
        """Get next rate limiter in round-robin fashion."""
        if not self._client_ids:
            raise ValueError("No rate limiters configured")
        
        client_id = self._client_ids[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self._client_ids)
        return self.limiters[client_id]
    
    async def acquire_any(
        self,
        estimated_input: int,
        estimated_output: int,
        timeout: float = 30.0
    ) -> Optional[ClientRateLimiter]:
        """
        Try to acquire rate limit from any available limiter.
        
        Returns:
            The limiter that was acquired, or None if timeout
        """
        start_time = time.time()
        attempts = 0
        
        while time.time() - start_time < timeout:
            # Try each limiter in round-robin order
            for _ in range(len(self._client_ids)):
                limiter = self.get_next_limiter()
                
                # Try with a short timeout
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    break
                
                acquired = await limiter.acquire_for_request(
                    estimated_input,
                    estimated_output,
                    min(1.0, remaining)  # Try for at most 1 second per limiter
                )
                
                if acquired:
                    return limiter
                
                attempts += 1
            
            # Small delay before retrying
            await asyncio.sleep(0.1)
        
        logger.warning(f"Could not acquire rate limit from pool after {attempts} attempts")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all limiters."""
        return {
            client_id: limiter.get_metrics()
            for client_id, limiter in self.limiters.items()
        }