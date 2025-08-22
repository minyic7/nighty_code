"""
Logging middleware for request/response tracking.
"""

import time
import json
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from .base import Middleware
from ..core.types import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """
    Middleware that logs requests, responses, and performance metrics.
    """
    
    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = True,
        log_errors: bool = True,
        include_content: bool = False,
        max_content_length: int = 500
    ):
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_errors = log_errors
        self.include_content = include_content
        self.max_content_length = max_content_length
    
    async def process_request(
        self,
        request: CompletionRequest,
        context: dict
    ) -> CompletionRequest:
        """Log incoming request."""
        context['request_start_time'] = time.time()
        context['request_id'] = self._generate_request_id()
        
        if self.log_requests:
            log_data = {
                'request_id': context['request_id'],
                'timestamp': datetime.now().isoformat(),
                'messages_count': len(request.messages) if request.messages else 0,
                'temperature': request.temperature,
                'max_tokens': request.max_tokens,
                'stream': request.stream,
            }
            
            if self.include_content and request.messages:
                # Truncate content if too long
                messages_preview = []
                for msg in request.messages[:3]:  # Only first 3 messages
                    content = msg.content
                    if len(content) > self.max_content_length:
                        content = content[:self.max_content_length] + "..."
                    messages_preview.append({
                        'role': msg.role.value,
                        'content': content
                    })
                log_data['messages_preview'] = messages_preview
            
            logger.info(f"LLM Request: {json.dumps(log_data, indent=2)}")
        
        return request
    
    async def process_response(
        self,
        response: CompletionResponse,
        context: dict
    ) -> CompletionResponse:
        """Log response and performance metrics."""
        if self.log_responses:
            duration = time.time() - context.get('request_start_time', time.time())
            
            log_data = {
                'request_id': context.get('request_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 3),
                'model': response.model,
                'provider': response.provider.value,
            }
            
            if response.usage:
                log_data['usage'] = response.usage
                
                # Calculate tokens per second
                if duration > 0:
                    total_tokens = response.usage.get('total_tokens', 0)
                    log_data['tokens_per_second'] = round(total_tokens / duration, 2)
            
            if self.include_content:
                content = response.content
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
                log_data['content_preview'] = content
            
            # Add retry information if available
            if context.get('retry_count', 0) > 0:
                log_data['retries'] = context['retry_count']
            
            logger.info(f"LLM Response: {json.dumps(log_data, indent=2)}")
        
        return response
    
    async def process_error(
        self,
        error: Exception,
        context: dict
    ) -> Optional[Exception]:
        """Log errors."""
        if self.log_errors:
            duration = time.time() - context.get('request_start_time', time.time())
            
            log_data = {
                'request_id': context.get('request_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 3),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'retry_count': context.get('retry_count', 0),
            }
            
            logger.error(f"LLM Error: {json.dumps(log_data, indent=2)}")
        
        return error
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        import uuid
        return f"req_{uuid.uuid4().hex[:12]}"