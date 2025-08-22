"""Utility functions for LLM module"""

from typing import TypeVar, Type
from contextlib import asynccontextmanager

T = TypeVar('T')


class AsyncContextManagerMixin:
    """Mixin class for async context manager support"""
    
    async def __aenter__(self: T) -> T:
        if hasattr(self, 'initialize'):
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'close'):
            await self.close()