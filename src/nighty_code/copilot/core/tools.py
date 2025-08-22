"""
Unified tool execution system with timeout protection.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    status: ExecutionStatus
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0


class ToolExecutor:
    """
    Simple tool executor with timeout protection and error handling.
    """
    
    def __init__(
        self,
        mcp_server,
        default_timeout: float = 5.0,
        max_retries: int = 2
    ):
        """
        Initialize tool executor.
        
        Args:
            mcp_server: MCP server instance
            default_timeout: Default timeout for tool execution
            max_retries: Maximum retry attempts for failed tools
        """
        self.mcp_server = mcp_server
        self.default_timeout = default_timeout
        self.max_retries = max_retries
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a single tool with timeout protection.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            timeout: Optional timeout override
            
        Returns:
            ToolResult with execution outcome
        """
        timeout = timeout or self.default_timeout
        start_time = asyncio.get_event_loop().time()
        
        for attempt in range(self.max_retries):
            try:
                # Create task for tool execution
                task = asyncio.create_task(
                    self._call_tool(tool_name, params)
                )
                
                # Wait with timeout
                try:
                    result_data = await asyncio.wait_for(task, timeout=timeout)
                    
                    return ToolResult(
                        tool_name=tool_name,
                        status=ExecutionStatus.SUCCESS,
                        data=result_data,
                        duration=asyncio.get_event_loop().time() - start_time
                    )
                    
                except asyncio.TimeoutError:
                    # Cancel the task
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Tool {tool_name} timed out, retrying...")
                        continue
                    
                    return ToolResult(
                        tool_name=tool_name,
                        status=ExecutionStatus.TIMEOUT,
                        error=f"Timeout after {timeout}s",
                        duration=asyncio.get_event_loop().time() - start_time
                    )
                    
            except Exception as e:
                logger.debug(f"Tool {tool_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                
                return ToolResult(
                    tool_name=tool_name,
                    status=ExecutionStatus.FAILED,
                    error=str(e),
                    duration=asyncio.get_event_loop().time() - start_time
                )
        
        # Should not reach here
        return ToolResult(
            tool_name=tool_name,
            status=ExecutionStatus.FAILED,
            error="Max retries exceeded",
            duration=asyncio.get_event_loop().time() - start_time
        )
    
    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call MCP server tool."""
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
        stop_on_failure: bool = False,
        parallel: bool = False
    ) -> List[ToolResult]:
        """
        Execute multiple tools.
        
        Args:
            tools: List of tool specifications with 'name', 'params', and optional 'timeout'
            stop_on_failure: Whether to stop execution on first failure
            parallel: Whether to execute tools in parallel
            
        Returns:
            List of ToolResults
        """
        if parallel:
            return await self._execute_parallel(tools)
        else:
            return await self._execute_sequential(tools, stop_on_failure)
    
    async def _execute_sequential(
        self,
        tools: List[Dict[str, Any]],
        stop_on_failure: bool
    ) -> List[ToolResult]:
        """Execute tools sequentially."""
        results = []
        
        for tool_spec in tools:
            result = await self.execute(
                tool_spec['name'],
                tool_spec.get('params', {}),
                tool_spec.get('timeout')
            )
            results.append(result)
            
            if stop_on_failure and result.status != ExecutionStatus.SUCCESS:
                # Add skipped results for remaining tools
                for remaining in tools[len(results):]:
                    results.append(ToolResult(
                        tool_name=remaining['name'],
                        status=ExecutionStatus.SKIPPED,
                        error="Skipped due to previous failure"
                    ))
                break
        
        return results
    
    async def _execute_parallel(self, tools: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tools in parallel."""
        tasks = []
        
        for tool_spec in tools:
            task = self.execute(
                tool_spec['name'],
                tool_spec.get('params', {}),
                tool_spec.get('timeout')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ToolResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_name=tools[i]['name'],
                    status=ExecutionStatus.FAILED,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def execute_from_intent(
        self,
        intent,
        context: Dict[str, Any]
    ) -> List[ToolResult]:
        """
        Execute tools based on intent.
        
        Args:
            intent: ProcessedIntent with tool suggestions
            context: Execution context
            
        Returns:
            List of ToolResults
        """
        # Build tool plan from intent
        tool_plan = self._build_tool_plan(intent, context)
        
        if not tool_plan:
            return []
        
        # Execute tools
        return await self.execute_multiple(tool_plan)
    
    def _build_tool_plan(self, intent, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build tool execution plan from intent."""
        plan = []
        
        # Map suggested tools to actual tool calls
        for tool_name in intent.suggested_tools:
            params = self._build_tool_params(tool_name, intent, context)
            
            if params is not None:
                plan.append({
                    'name': tool_name,
                    'params': params,
                    'timeout': self.default_timeout
                })
        
        return plan
    
    def _build_tool_params(
        self,
        tool_name: str,
        intent,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build parameters for a specific tool."""
        # Extract relevant entities
        from .intent import EntityType
        
        if tool_name == 'read_file':
            # Find file entity
            for entity in intent.entities:
                if entity.type in [EntityType.FILE_PATH, EntityType.FILE_NAME]:
                    return {'file_path': entity.value}
            return None
            
        elif tool_name == 'list_directory':
            # Find directory entity or use current
            for entity in intent.entities:
                if entity.type == EntityType.DIR_PATH:
                    return {'directory_path': entity.value}
            return {'directory_path': context.get('current_directory', '.')}
            
        elif tool_name == 'search_pattern':
            # Find pattern entity
            for entity in intent.entities:
                if entity.type == EntityType.CODE_PATTERN:
                    return {'pattern': entity.value}
            # Fallback to keywords
            if intent.keywords:
                return {'pattern': intent.keywords[0]}
            return None
            
        elif tool_name == 'search_files':
            # Use first entity or keyword
            if intent.entities:
                return {'query': intent.entities[0].value}
            elif intent.keywords:
                return {'query': intent.keywords[0]}
            return None
            
        elif tool_name == 'fuzzy_find':
            # Similar to search_files
            if intent.entities:
                return {'query': intent.entities[0].value}
            elif intent.keywords:
                return {'query': intent.keywords[0]}
            return None
            
        else:
            # Unknown tool, skip
            logger.warning(f"Unknown tool: {tool_name}")
            return None