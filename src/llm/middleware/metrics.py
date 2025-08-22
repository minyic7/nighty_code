"""
Metrics collection middleware for monitoring and observability.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .base import Middleware
from ..core.types import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Container for collected metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    total_latency_seconds: float = 0.0
    min_latency_seconds: float = float('inf')
    max_latency_seconds: float = 0.0
    
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_by_model: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_seconds / self.successful_requests
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.get_success_rate(),
            'total_retries': self.total_retries,
            
            'tokens': {
                'total_input': self.total_input_tokens,
                'total_output': self.total_output_tokens,
                'total': self.total_tokens,
            },
            
            'latency': {
                'average_seconds': self.get_average_latency(),
                'min_seconds': self.min_latency_seconds if self.min_latency_seconds != float('inf') else 0,
                'max_seconds': self.max_latency_seconds,
                'total_seconds': self.total_latency_seconds,
            },
            
            'errors_by_type': dict(self.errors_by_type),
            'requests_by_model': dict(self.requests_by_model),
            'tokens_by_model': dict(self.tokens_by_model),
        }


class MetricsMiddleware(Middleware):
    """
    Middleware that collects metrics for monitoring and analysis.
    """
    
    def __init__(self, enable_per_provider_metrics: bool = True):
        self.metrics = Metrics()
        self.enable_per_provider_metrics = enable_per_provider_metrics
        
        # Per-provider metrics
        self.provider_metrics: Dict[str, Metrics] = defaultdict(Metrics)
    
    async def process_request(
        self,
        request: CompletionRequest,
        context: dict
    ) -> CompletionRequest:
        """Track request start."""
        context['metrics_start_time'] = time.time()
        
        self.metrics.total_requests += 1
        
        # Track retries
        if context.get('retry_count', 0) > 0:
            self.metrics.total_retries += 1
        
        return request
    
    async def process_response(
        self,
        response: CompletionResponse,
        context: dict
    ) -> CompletionResponse:
        """Track successful response metrics."""
        duration = time.time() - context.get('metrics_start_time', time.time())
        
        # Update global metrics
        self.metrics.successful_requests += 1
        self.metrics.total_latency_seconds += duration
        self.metrics.min_latency_seconds = min(self.metrics.min_latency_seconds, duration)
        self.metrics.max_latency_seconds = max(self.metrics.max_latency_seconds, duration)
        
        # Track model usage
        self.metrics.requests_by_model[response.model] += 1
        
        # Track token usage
        if response.usage:
            input_tokens = response.usage.get('prompt_tokens', 0)
            output_tokens = response.usage.get('completion_tokens', 0)
            total_tokens = response.usage.get('total_tokens', 0)
            
            self.metrics.total_input_tokens += input_tokens
            self.metrics.total_output_tokens += output_tokens
            self.metrics.total_tokens += total_tokens
            self.metrics.tokens_by_model[response.model] += total_tokens
        
        # Track per-provider metrics
        if self.enable_per_provider_metrics:
            provider_key = response.provider.value
            provider_metric = self.provider_metrics[provider_key]
            
            provider_metric.successful_requests += 1
            provider_metric.total_requests += 1
            provider_metric.total_latency_seconds += duration
            provider_metric.min_latency_seconds = min(provider_metric.min_latency_seconds, duration)
            provider_metric.max_latency_seconds = max(provider_metric.max_latency_seconds, duration)
            
            if response.usage:
                provider_metric.total_input_tokens += input_tokens
                provider_metric.total_output_tokens += output_tokens
                provider_metric.total_tokens += total_tokens
        
        return response
    
    async def process_error(
        self,
        error: Exception,
        context: dict
    ) -> Optional[Exception]:
        """Track error metrics."""
        self.metrics.failed_requests += 1
        self.metrics.errors_by_type[type(error).__name__] += 1
        
        return error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        result = {
            'global': self.metrics.to_dict()
        }
        
        if self.enable_per_provider_metrics:
            result['by_provider'] = {
                provider: metrics.to_dict()
                for provider, metrics in self.provider_metrics.items()
            }
        
        return result
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = Metrics()
        self.provider_metrics.clear()
        logger.info("Metrics reset")