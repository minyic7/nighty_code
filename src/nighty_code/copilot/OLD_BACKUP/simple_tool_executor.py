"""
Simple tool executor that actually respects timeouts.
No fancy recovery, just reliable execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Simple execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Simple tool result."""
    tool_name: str
    status: ExecutionStatus
    data: Any = None
    error: Optional[str] = None


class SimpleToolExecutor:
    """
    Simple tool executor that works.
    Handles timeouts properly, no overengineering.
    """
    
    def __init__(self, mcp_server, default_timeout: float = 5.0):
        self.mcp_server = mcp_server
        self.default_timeout = default_timeout
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a single tool with proper timeout.
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Create the task
            task = asyncio.create_task(
                self._call_tool(tool_name, params)
            )
            
            # Wait with timeout
            try:
                result_data = await asyncio.wait_for(task, timeout=timeout)
                return ToolResult(
                    tool_name=tool_name,
                    status=ExecutionStatus.SUCCESS,
                    data=result_data
                )
            except asyncio.TimeoutError:
                # Cancel the task
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                return ToolResult(
                    tool_name=tool_name,
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Timeout after {timeout}s"
                )
                
        except Exception as e:
            logger.debug(f"Tool {tool_name} failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call MCP tool."""
        response = await self.mcp_server.handle_request({
            'method': f'tool/{tool_name}',
            'params': params
        })
        
        if 'error' in response:
            raise Exception(response['error'])
        
        return response.get('result')
    
    async def execute_multiple(
        self,
        tools: List[Dict[str, Any]],
        stop_on_failure: bool = False
    ) -> List[ToolResult]:
        """
        Execute multiple tools in sequence.
        Simple, reliable, no magic.
        """
        results = []
        
        for tool_spec in tools:
            result = await self.execute(
                tool_spec['name'],
                tool_spec.get('params', {}),
                tool_spec.get('timeout')
            )
            results.append(result)
            
            if stop_on_failure and result.status != ExecutionStatus.SUCCESS:
                break
        
        return results