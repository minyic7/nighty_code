# src/llm/core/pool.py
import asyncio
from typing import Dict, List, Optional, Set, Union
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta

from .types import LLMProvider, LLMConfig, ClientStatus
from .config import PoolConfig
from .exceptions import (
    LLMPoolExhaustedError,
    LLMException,
    LLMConfigurationError
)
from ..providers.base import BaseLLMProvider
from ..utils import AsyncContextManagerMixin


logger = logging.getLogger(__name__)


class LLMConnectionPool(AsyncContextManagerMixin):
    """Connection pool for LLM providers"""

    def __init__(self, configs: Union[LLMConfig, List[LLMConfig]], pool_config: Optional[PoolConfig] = None):
        # Support both single config (backward compat) and multiple configs
        if isinstance(configs, LLMConfig):
            configs = [configs]
        
        self.configs = configs
        self.config = configs[0] if configs else None  # Keep for backward compatibility
        self.pool_config = pool_config or PoolConfig()

        # Pool state
        self._clients: List[BaseLLMProvider] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: Set[BaseLLMProvider] = set()
        self._lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        
        # Round-robin counter for cycling through configs
        self._config_index = 0

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "failed_requests": 0,
            "pool_exhausted_count": 0,
            "average_wait_time": 0.0,
        }

    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            provider_name = self.configs[0].provider.value if self.configs else "unknown"
            logger.info(f"Initializing pool for {provider_name} with {len(self.configs)} configurations")

            # Create minimum number of clients, cycling through configs
            for _ in range(self.pool_config.min_size):
                await self._create_client()

            # Start health check task
            if self.pool_config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

            self._initialized = True
            provider_name = self.configs[0].provider.value if self.configs else "unknown"
            logger.info(
                f"Pool initialized with {len(self._clients)} clients "
                f"for {provider_name}"
            )

    async def _create_client(self) -> BaseLLMProvider:
        """Create a new client and add it to the pool"""
        # Import here to avoid circular dependency
        from ..providers import get_provider
        
        # Get the next config in round-robin fashion
        if not self.configs:
            raise LLMConfigurationError("No configurations available")
        
        config = self.configs[self._config_index]
        self._config_index = (self._config_index + 1) % len(self.configs)
        
        provider_class = get_provider(config.provider.value)
        client = provider_class(config, config.rate_limit_config)
        await client.initialize()
        
        logger.debug(f"Created new client {client._client_id} with API key index {(self._config_index - 1) % len(self.configs)}")

        self._clients.append(client)
        await self._available.put(client)
        
        return client

    async def acquire(self, timeout: Optional[float] = None) -> BaseLLMProvider:
        """Acquire a client from the pool"""
        if self._closed:
            raise LLMException("Pool is closed")

        if not self._initialized:
            await self.initialize()

        timeout = timeout or self.pool_config.acquire_timeout
        start_time = datetime.now()

        while True:
            try:
                # Calculate remaining timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining_timeout = timeout - elapsed if timeout else None
                
                if remaining_timeout is not None and remaining_timeout <= 0:
                    # Timeout exceeded
                    self._metrics["pool_exhausted_count"] += 1
                    raise LLMPoolExhaustedError(
                        f"Could not acquire client within {timeout} seconds"
                    )
                
                # Try to get an available client with remaining timeout
                try:
                    wait_timeout = min(1.0, remaining_timeout) if remaining_timeout else 1.0
                    logger.debug(f"Waiting for available client with timeout {wait_timeout}s (queue size: {self._available.qsize()})")
                    client = await asyncio.wait_for(
                        self._available.get(), 
                        timeout=wait_timeout
                    )
                    logger.debug(f"Got client {client._client_id} from queue")
                except asyncio.TimeoutError:
                    # Check if we should expand the pool
                    async with self._lock:
                        # Only expand if we haven't reached the limit of configured API keys
                        # and we're below max_size
                        if (len(self._clients) < len(self.configs) and 
                            len(self._clients) < self.pool_config.max_size):
                            logger.info(f"Expanding pool: {len(self._clients)} -> {len(self._clients) + 1}")
                            client = await self._create_client()
                            # The newly created client is already in the queue, get it
                            client = await self._available.get()
                        else:
                            # Can't expand, continue waiting
                            continue

                # Check if client is still healthy
                if not client._status.is_available:
                    logger.warning(f"Client {client._client_id} is not available")
                    # Remove unhealthy client
                    await self._remove_client(client)
                    
                    # Try to create a replacement if we're below min size
                    async with self._lock:
                        if len(self._clients) < self.pool_config.min_size:
                            await self._create_client()
                    
                    # Continue loop to get another client
                    continue

                # Mark as in use
                self._in_use.add(client)
                await client.acquire()

                # Update metrics
                wait_time = (datetime.now() - start_time).total_seconds()
                self._update_average_wait_time(wait_time)
                self._metrics["total_requests"] += 1
                
                logger.debug(f"Acquired client {client._client_id} (queue size now: {self._available.qsize()}, in_use: {len(self._in_use)})")

                return client

            except LLMPoolExhaustedError:
                raise
            except Exception as e:
                logger.error(f"Error acquiring client: {e}")
                raise LLMException(f"Failed to acquire client: {e}")

    async def release(self, client: BaseLLMProvider):
        """Release a client back to the pool"""
        if client not in self._in_use:
            logger.warning(f"Client {client._client_id} was not in use")
            return

        try:
            await client.release()
            self._in_use.remove(client)

            # Check if client is still healthy
            if client._status.is_available and not self._closed:
                # Add a small delay to allow other clients to be picked up
                # This helps with round-robin distribution
                await asyncio.sleep(0.001)  # 1ms delay
                await self._available.put(client)
                logger.debug(f"Released client {client._client_id} back to pool (queue size now: {self._available.qsize()})")
            else:
                logger.warning(
                    f"Client {client._client_id} is not healthy, removing from pool"
                )
                await self._remove_client(client)

                # Create a replacement if needed
                async with self._lock:
                    if len(self._clients) < self.pool_config.min_size:
                        await self._create_client()

        except Exception as e:
            logger.error(f"Error releasing client {client._client_id}: {e}")

    async def _remove_client(self, client: BaseLLMProvider):
        """Remove a client from the pool"""
        try:
            await client.close()
            if client in self._clients:
                self._clients.remove(client)
            if client in self._in_use:
                self._in_use.remove(client)
        except Exception as e:
            logger.error(f"Error removing client {client._client_id}: {e}")

    @asynccontextmanager
    async def get_client(self, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing a client"""
        client = None
        try:
            logger.debug(f"Attempting to acquire client from pool (available: {self._available.qsize()}, in_use: {len(self._in_use)})")
            client = await self.acquire(timeout)
            logger.debug(f"Acquired client {client._client_id}")
            yield client
        finally:
            if client:
                logger.debug(f"Releasing client {client._client_id}")
                await self.release(client)
                logger.debug(f"Released client {client._client_id} (available: {self._available.qsize()}, in_use: {len(self._in_use)})")

    async def _health_check_loop(self):
        """Periodically check the health of clients in the pool"""
        while not self._closed:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval)

                # Check health of all clients
                for client in self._clients[:]:  # Copy list to avoid modification during iteration
                    if client not in self._in_use:
                        if not await client.health_check():
                            logger.warning(
                                f"Client {client._client_id} failed health check"
                            )
                            await self._remove_client(client)

                # Ensure minimum pool size
                async with self._lock:
                    while len(self._clients) < self.pool_config.min_size:
                        await self._create_client()

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    def _update_average_wait_time(self, wait_time: float):
        """Update the average wait time metric"""
        current_avg = self._metrics["average_wait_time"]
        total_requests = self._metrics["total_requests"]

        if total_requests == 0:
            self._metrics["average_wait_time"] = wait_time
        else:
            self._metrics["average_wait_time"] = (
                (current_avg * (total_requests - 1) + wait_time) / total_requests
            )

    async def close(self):
        """Close the pool and all clients"""
        if self._closed:
            return

        self._closed = True

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all clients
        for client in self._clients:
            await client.close()

        self._clients.clear()
        self._in_use.clear()

        # Clear the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

        provider_name = self.configs[0].provider.value if self.configs else "unknown"
        logger.info(f"Pool for {provider_name} closed")

    def get_status(self) -> Dict:
        """Get the current status of the pool"""
        provider_name = self.configs[0].provider.value if self.configs else "unknown"
        return {
            "provider": provider_name,
            "total_configs": len(self.configs),
            "total_clients": len(self._clients),
            "available_clients": self._available.qsize(),
            "in_use_clients": len(self._in_use),
            "closed": self._closed,
            "metrics": self._metrics.copy(),
            "clients": [client._status.__dict__ for client in self._clients]
        }

    # Context manager methods inherited from AsyncContextManagerMixin