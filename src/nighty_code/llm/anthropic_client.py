"""
Anthropic Claude API client implementation.
"""

import json
import logging
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Client for Anthropic Claude API."""
    
    API_BASE = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"
    
    def __init__(self, api_key: str):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json"
        }
    
    def create_message(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a message using Claude API.
        
        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system: System prompt (optional)
            **kwargs: Additional parameters
            
        Returns:
            API response as dictionary
        """
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add system prompt if provided
        if system:
            payload["system"] = system
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Make API request
        url = f"{self.API_BASE}/messages"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"API error details: {error_data}")
                    raise Exception(f"Anthropic API error: {error_data.get('error', {}).get('message', str(e))}")
                except:
                    raise Exception(f"Anthropic API error: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Claude uses a different tokenizer than OpenAI, but as a rough estimate:
        - 1 token ≈ 4 characters for English text
        - 1 token ≈ 2-3 characters for code
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation - Claude's actual tokenizer is not publicly available
        # This is conservative to avoid exceeding limits
        return len(text) // 3
    
    @staticmethod
    def format_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Anthropic response to match our standard format.
        
        Args:
            response: Raw API response
            
        Returns:
            Standardized response format
        """
        
        # Extract content from response
        content = ""
        if "content" in response and response["content"]:
            # Claude returns content as a list of content blocks
            for block in response["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
        
        # Extract usage information
        usage = response.get("usage", {})
        
        return {
            "content": content,
            "model": response.get("model", ""),
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            },
            "stop_reason": response.get("stop_reason", ""),
            "id": response.get("id", "")
        }