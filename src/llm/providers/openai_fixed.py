# src/llm/providers/openai_fixed.py
"""
Fixed OpenAI provider with better validation
"""

from typing import AsyncIterator, Optional
import logging
import uuid

from ..core.base import BaseLLMProvider
from ..core.types import (
    CompletionRequest,
    CompletionResponse,
    ClientStatus,
    LLMConfig,
    LLMProvider,
)
from ..core.exceptions import (
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation with fixed validation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client: Optional[Any] = None
        self._client_id = f"openai_{uuid.uuid4().hex[:8]}"
        self._status = ClientStatus(
            client_id=self._client_id,
            provider=LLMProvider.OPENAI,
            model=config.model,
            is_available=False,
            in_use=False,
        )
    
    async def _initialize_provider(self):
        """Initialize the OpenAI client"""
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Validate the connection with the FIXED validation method
            if await self.validate_connection():
                self._status.is_available = True
                logger.info(f"OpenAI client {self._client_id} initialized successfully")
            else:
                raise LLMProviderError("Failed to validate OpenAI connection")
                
        except ImportError:
            raise LLMProviderError(
                "OpenAI library not installed. Run: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise LLMProviderError(f"Failed to initialize OpenAI client: {e}")
    
    async def validate_connection(self) -> bool:
        """
        FIXED: Validate connection using models.list() instead of models.retrieve()
        This works more reliably across different OpenAI API configurations
        """
        try:
            # Method 1: Try the original validation first
            try:
                response = await self.client.models.retrieve(self.config.model)
                if response.id == self.config.model:
                    logger.info(f"Validation successful using models.retrieve()")
                    return True
            except Exception as e:
                logger.debug(f"models.retrieve() failed: {e}, trying models.list()")
            
            # Method 2: Fallback to listing models and checking if our model exists
            models_response = await self.client.models.list()
            available_models = [model.id for model in models_response.data]
            
            if self.config.model in available_models:
                logger.info(f"Validation successful: Model '{self.config.model}' found in available models")
                return True
            else:
                logger.warning(f"Model '{self.config.model}' not found in available models: {available_models[:5]}...")
                return False
                
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            return False
    
    # ... rest of the implementation stays the same ...