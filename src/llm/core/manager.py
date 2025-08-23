# src/llm/core/manager.py
from typing import Dict, Optional, List
import asyncio
import logging

from .types import LLMProvider, LLMConfig
from .config import PoolConfig, ConfigManager
from .pool import LLMConnectionPool
from .client import LLMClient
from .exceptions import LLMConfigurationError, LLMException
from ..utils import AsyncContextManagerMixin


logger = logging.getLogger(__name__)


class LLMManager(AsyncContextManagerMixin):
    """Central manager for all LLM operations"""
    
    _instance: Optional['LLMManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._pools: Dict[LLMProvider, LLMConnectionPool] = {}
        self._clients: Dict[LLMProvider, LLMClient] = {}
        self._config_manager = ConfigManager()
        self._initialized = False
    
    @classmethod
    async def get_instance(cls) -> 'LLMManager':
        """Get or create the singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self):
        """Initialize the manager with configured providers"""
        if self._initialized:
            return
        
        logger.info("Initializing LLM Manager")
        
        # Initialize pools for all configured providers
        for provider in self._config_manager.global_config.providers.keys():
            try:
                await self._create_pool(provider)
            except Exception as e:
                logger.error(f"Failed to initialize pool for {provider}: {e}")
        
        self._initialized = True
        logger.info(f"LLM Manager initialized with {len(self._pools)} providers")
    
    async def _create_pool(
        self,
        provider: LLMProvider,
        configs: List[LLMConfig] = None,
        pool_config: Optional[PoolConfig] = None
    ) -> LLMConnectionPool:
        """Create a new connection pool for a provider"""
        pool_config = pool_config or self._config_manager.global_config.pool_config
        
        # Get all configs for this provider if not provided
        if configs is None:
            configs = self._config_manager.get_all_provider_configs(provider)
        
        if not configs:
            # Fallback to single config for backward compatibility
            single_config = self._config_manager.get_provider_config(provider)
            if single_config:
                configs = [single_config]
            else:
                raise LLMConfigurationError(f"No configurations found for {provider}")
        
        pool = LLMConnectionPool(configs, pool_config)
        await pool.initialize()
        
        self._pools[provider] = pool
        self._clients[provider] = LLMClient(pool, self._config_manager.global_config.instructor_config)
        
        logger.info(f"Created pool for {provider.value} with {len(configs)} configurations")
        return pool
    
    async def add_provider(
        self,
        config: LLMConfig,
        pool_config: Optional[PoolConfig] = None
    ):
        """Add a new provider configuration"""
        self._config_manager.add_provider(config)
        
        # If already initialized, create the pool immediately
        if self._initialized:
            configs = self._config_manager.get_all_provider_configs(config.provider)
            await self._create_pool(config.provider, configs, pool_config)
    
    def get_client(
        self,
        provider: Optional[LLMProvider] = None
    ) -> LLMClient:
        """Get a client for the specified provider"""
        if not self._initialized:
            raise LLMException("Manager not initialized")
        
        # Use default provider if none specified
        if provider is None:
            provider = self._config_manager.global_config.default_provider
            if provider is None:
                raise LLMConfigurationError("No default provider configured")
        
        if provider not in self._clients:
            raise LLMConfigurationError(f"Provider {provider} not configured")
        
        return self._clients[provider]
    
    async def get_or_create_client(
        self,
        provider: LLMProvider,
        configs: Optional[List[LLMConfig]] = None,
        pool_config: Optional[PoolConfig] = None
    ) -> LLMClient:
        """Get an existing client or create a new one"""
        if provider in self._clients:
            return self._clients[provider]
        
        if configs is None:
            configs = self._config_manager.get_all_provider_configs(provider)
            if not configs:
                # Try single config for backward compatibility
                single_config = self._config_manager.get_provider_config(provider)
                if single_config:
                    configs = [single_config]
                else:
                    raise LLMConfigurationError(
                        f"No configuration found for {provider}"
                    )
        
        await self._create_pool(provider, configs, pool_config)
        return self._clients[provider]
    
    def list_providers(self) -> List[LLMProvider]:
        """List all configured providers"""
        return list(self._clients.keys())
    
    def get_status(self, provider: Optional[LLMProvider] = None) -> Dict:
        """Get status of specific provider or all providers"""
        if provider:
            if provider not in self._pools:
                raise LLMConfigurationError(f"Provider {provider} not configured")
            return self._pools[provider].get_status()
        
        # Return all provider statuses
        return {
            provider.value: pool.get_status()
            for provider, pool in self._pools.items()
        }
    
    async def close_provider(self, provider: LLMProvider):
        """Close a specific provider's pool"""
        if provider in self._pools:
            await self._pools[provider].close()
            del self._pools[provider]
            del self._clients[provider]
            logger.info(f"Closed provider {provider.value}")
    
    async def close(self):
        """Close all pools and cleanup"""
        logger.info("Closing LLM Manager")
        
        # Close all pools
        tasks = [pool.close() for pool in self._pools.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._pools.clear()
        self._clients.clear()
        self._initialized = False
        
        logger.info("LLM Manager closed")
    
    # Context manager methods inherited from AsyncContextManagerMixin


# Convenience function for getting the global manager
async def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    return await LLMManager.get_instance()