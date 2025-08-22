"""
Production-ready tool chain with comprehensive error handling,
concurrency control, and recovery strategies.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
import json

from .hybrid_intent import ProcessedIntent, IntentToToolMapper

logger = logging.getLogger(__name__)


@dataclass
class ToolChainConfig:
    """Configuration for production tool chain."""
    max_retries: int = 3
    default_timeout: float = 15.0
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_concurrent: int = 5
    enable_recovery: bool = True
    enable_rollback: bool = True
    enable_validation: bool = True


class ToolStatus(Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RECOVERED = "recovered"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    status: ToolStatus
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    recovered_by: Optional[str] = None


@dataclass
class ExecutionContext:
    """Context for tool execution."""
    request_id: str
    intent: ProcessedIntent
    user_context: Dict[str, Any]
    results: List[ToolResult] = field(default_factory=list)
    rollback_stack: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class ConcurrencyController:
    """Control concurrent tool execution."""
    
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running_tools: Set[str] = set()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tool_name: str) -> bool:
        """Acquire permission to run tool."""
        async with self.lock:
            # Check for conflicts
            if self._has_conflict(tool_name):
                return False
            
            await self.semaphore.acquire()
            self.running_tools.add(tool_name)
            return True
    
    async def release(self, tool_name: str):
        """Release tool execution slot."""
        async with self.lock:
            if tool_name in self.running_tools:
                self.running_tools.remove(tool_name)
                self.semaphore.release()
    
    def _has_conflict(self, tool_name: str) -> bool:
        """Check if tool conflicts with running tools."""
        # Define conflicting tool pairs
        conflicts = {
            'read_file': {'write_file', 'delete_file'},
            'write_file': {'read_file', 'delete_file'},
            'delete_file': {'read_file', 'write_file'}
        }
        
        if tool_name in conflicts:
            return bool(conflicts[tool_name] & self.running_tools)
        return False


class RecoveryStrategy:
    """Recovery strategies for failed tools."""
    
    def __init__(self):
        self.strategies = {
            'file_not_found': self._recover_file_not_found,
            'permission_denied': self._recover_permission_denied,
            'timeout': self._recover_timeout,
            'network_error': self._recover_network_error,
            'generic': self._recover_generic
        }
    
    def get_recovery_tool(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Get recovery tool for error."""
        error_type = self._classify_error(error)
        if error_type in self.strategies:
            return self.strategies[error_type](tool_name, error)
        return self.strategies['generic'](tool_name, error)
    
    def _classify_error(self, error: str) -> str:
        """Classify error type."""
        error_lower = error.lower()
        if 'not found' in error_lower or 'no such file' in error_lower:
            return 'file_not_found'
        elif 'permission' in error_lower or 'access denied' in error_lower:
            return 'permission_denied'
        elif 'timeout' in error_lower:
            return 'timeout'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network_error'
        return 'generic'
    
    def _recover_file_not_found(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Recovery for file not found."""
        if tool_name == 'read_file':
            return {
                'name': 'fuzzy_find',
                'params': {'query': self._extract_filename(error)},
                'description': 'Finding similar files'
            }
        return None
    
    def _recover_permission_denied(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Recovery for permission errors."""
        # Try alternative approach
        if tool_name == 'read_file':
            return {
                'name': 'list_directory',
                'params': {'directory_path': '.'},
                'description': 'Listing accessible files'
            }
        return None
    
    def _recover_timeout(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Recovery for timeout errors."""
        # Retry with longer timeout
        return {
            'name': tool_name,
            'params': {},  # Will be filled from original
            'description': 'Retrying with longer timeout',
            'timeout': 30.0  # Double timeout
        }
    
    def _recover_network_error(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Recovery for network errors."""
        # Use cached or offline alternative
        return {
            'name': 'smart_suggest',
            'params': {'use_cache': True},
            'description': 'Using cached suggestions'
        }
    
    def _recover_generic(self, tool_name: str, error: str) -> Optional[Dict[str, Any]]:
        """Generic recovery strategy."""
        return {
            'name': 'smart_suggest',
            'params': {'context': {'failed_tool': tool_name, 'error': error}},
            'description': 'Getting alternative suggestions'
        }
    
    def _extract_filename(self, error: str) -> str:
        """Extract filename from error message."""
        # Simple extraction - could be improved
        parts = error.split("'")
        if len(parts) > 1:
            return parts[1].split('/')[-1].split('\\')[-1]
        return "unknown"


class RollbackManager:
    """Manage rollback of tool operations."""
    
    def __init__(self):
        self.rollback_handlers = {
            'write_file': self._rollback_write_file,
            'delete_file': self._rollback_delete_file,
            'create_directory': self._rollback_create_directory
        }
    
    def record_operation(self, tool_name: str, params: Dict[str, Any], 
                        previous_state: Optional[Any] = None) -> Dict[str, Any]:
        """Record operation for potential rollback."""
        return {
            'tool_name': tool_name,
            'params': params,
            'previous_state': previous_state,
            'timestamp': time.time()
        }
    
    async def rollback(self, operation: Dict[str, Any], mcp_server) -> bool:
        """Rollback an operation."""
        tool_name = operation['tool_name']
        if tool_name in self.rollback_handlers:
            try:
                return await self.rollback_handlers[tool_name](operation, mcp_server)
            except Exception as e:
                logger.error(f"Rollback failed for {tool_name}: {e}")
                return False
        return True  # No rollback needed
    
    async def _rollback_write_file(self, operation: Dict[str, Any], mcp_server) -> bool:
        """Rollback file write."""
        if operation.get('previous_state'):
            # Restore previous content
            params = {
                'file_path': operation['params']['file_path'],
                'content': operation['previous_state']
            }
            result = await mcp_server.handle_request({
                'method': 'tool/write_file',
                'params': params
            })
            return 'error' not in result
        else:
            # Delete created file
            params = {'file_path': operation['params']['file_path']}
            result = await mcp_server.handle_request({
                'method': 'tool/delete_file',
                'params': params
            })
            return 'error' not in result
    
    async def _rollback_delete_file(self, operation: Dict[str, Any], mcp_server) -> bool:
        """Rollback file deletion."""
        if operation.get('previous_state'):
            # Restore deleted file
            params = {
                'file_path': operation['params']['file_path'],
                'content': operation['previous_state']
            }
            result = await mcp_server.handle_request({
                'method': 'tool/write_file',
                'params': params
            })
            return 'error' not in result
        return False
    
    async def _rollback_create_directory(self, operation: Dict[str, Any], mcp_server) -> bool:
        """Rollback directory creation."""
        # Remove created directory
        params = {'directory_path': operation['params']['directory_path']}
        result = await mcp_server.handle_request({
            'method': 'tool/delete_directory',
            'params': params
        })
        return 'error' not in result


class RobustToolChain:
    """
    Production-ready tool chain with comprehensive error handling,
    concurrency control, and recovery strategies.
    """
    
    def __init__(
        self,
        mcp_server,
        config: Optional[ToolChainConfig] = None,
        metrics=None,
        audit=None
    ):
        self.mcp_server = mcp_server
        self.config = config or ToolChainConfig()
        self.metrics = metrics
        self.audit = audit
        
        # Initialize components
        self.concurrency = ConcurrencyController(self.config.max_concurrent)
        self.recovery = RecoveryStrategy()
        self.rollback = RollbackManager()
        self.tool_mapper = IntentToToolMapper(mcp_server)
        
        # Track execution statistics
        self.execution_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'failed': 0,
            'recovered': 0,
            'average_time': 0.0
        })
    
    async def execute_from_intent(
        self,
        intent: ProcessedIntent,
        context: Dict[str, Any],
        request_id: str
    ) -> List[ToolResult]:
        """
        Execute tools based on intent with full production safeguards.
        
        Args:
            intent: Processed intent
            context: Execution context
            request_id: Request ID for tracing
            
        Returns:
            List of tool results
        """
        # Create execution context
        exec_context = ExecutionContext(
            request_id=request_id,
            intent=intent,
            user_context=context
        )
        
        try:
            # Map intent to tool plan
            tool_plan = self.tool_mapper.map(intent, context)
            
            # Validate tool plan
            if self.config.enable_validation:
                if not self._validate_tool_plan(tool_plan):
                    logger.warning(f"Invalid tool plan for {request_id}")
                    return [ToolResult(
                        tool_name="validation",
                        status=ToolStatus.FAILED,
                        success=False,
                        error="Invalid tool plan"
                    )]
            
            # Execute tools
            results = []
            for tool_spec in tool_plan.tools:
                # Check dependencies
                if 'depends_on' in tool_spec:
                    dep_idx = tool_spec['depends_on']
                    if dep_idx < len(results) and not results[dep_idx].success:
                        # Skip if dependency failed
                        results.append(ToolResult(
                            tool_name=tool_spec['name'],
                            status=ToolStatus.SKIPPED,
                            success=False,
                            error="Dependency failed"
                        ))
                        continue
                
                # Execute tool
                result = await self._execute_tool_safe(tool_spec, exec_context)
                results.append(result)
                exec_context.results.append(result)
                
                # Audit logging
                if self.audit:
                    self.audit.log_tool_execution(
                        request_id,
                        tool_spec['name'],
                        tool_spec.get('params', {}),
                        result.success
                    )
                
                # Stop on critical failure
                if not result.success and tool_spec.get('critical', False):
                    logger.warning(f"Critical tool {tool_spec['name']} failed, stopping execution")
                    break
            
            # Update statistics
            self._update_statistics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Tool chain execution failed for {request_id}: {e}")
            return [ToolResult(
                tool_name="tool_chain",
                status=ToolStatus.FAILED,
                success=False,
                error=str(e)
            )]
    
    async def _execute_tool_safe(
        self,
        tool_spec: Dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult:
        """Execute single tool with all safeguards."""
        tool_name = tool_spec['name']
        params = tool_spec.get('params', {})
        timeout = tool_spec.get('timeout', self.config.default_timeout)
        
        # Acquire concurrency slot
        if not await self.concurrency.acquire(tool_name):
            await asyncio.sleep(1)  # Wait briefly
            if not await self.concurrency.acquire(tool_name):
                return ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.SKIPPED,
                    success=False,
                    error="Concurrency conflict"
                )
        
        try:
            # Execute with retries
            retry_count = 0
            last_error = None
            
            while retry_count <= self.config.max_retries:
                try:
                    # Record for potential rollback
                    if self.config.enable_rollback:
                        previous_state = await self._get_previous_state(tool_name, params)
                        operation = self.rollback.record_operation(
                            tool_name, params, previous_state
                        )
                        context.rollback_stack.append(operation)
                    
                    # Execute tool with timeout
                    start_time = time.time()
                    async with asyncio.timeout(timeout):
                        response = await self.mcp_server.handle_request({
                            'method': f'tool/{tool_name}',
                            'params': self._resolve_params(params, context)
                        })
                    
                    execution_time = time.time() - start_time
                    
                    # Check response
                    if 'error' in response:
                        raise Exception(response['error'])
                    
                    # Success
                    return ToolResult(
                        tool_name=tool_name,
                        status=ToolStatus.SUCCESS,
                        success=True,
                        data=response.get('result'),
                        execution_time=execution_time,
                        retry_count=retry_count
                    )
                    
                except asyncio.TimeoutError:
                    last_error = f"Timeout after {timeout}s"
                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        await self._wait_before_retry(retry_count)
                        timeout *= 1.5  # Increase timeout
                        
                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        await self._wait_before_retry(retry_count)
            
            # All retries failed - try recovery
            if self.config.enable_recovery:
                recovery_result = await self._attempt_recovery(
                    tool_name, params, last_error, context
                )
                if recovery_result:
                    return recovery_result
            
            # Complete failure
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                success=False,
                error=last_error,
                retry_count=retry_count
            )
            
        finally:
            # Release concurrency slot
            await self.concurrency.release(tool_name)
    
    async def _attempt_recovery(
        self,
        tool_name: str,
        params: Dict[str, Any],
        error: str,
        context: ExecutionContext
    ) -> Optional[ToolResult]:
        """Attempt recovery for failed tool."""
        recovery_tool = self.recovery.get_recovery_tool(tool_name, error)
        if not recovery_tool:
            return None
        
        logger.info(f"Attempting recovery for {tool_name} with {recovery_tool['name']}")
        
        # Update params if needed
        if not recovery_tool.get('params'):
            recovery_tool['params'] = params
        
        # Execute recovery tool
        result = await self._execute_tool_safe(recovery_tool, context)
        
        if result.success:
            result.status = ToolStatus.RECOVERED
            result.recovered_by = recovery_tool['name']
        
        return result
    
    async def _get_previous_state(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get previous state for rollback."""
        if tool_name == 'write_file':
            # Get current file content
            try:
                response = await self.mcp_server.handle_request({
                    'method': 'tool/read_file',
                    'params': {'file_path': params['file_path']}
                })
                if 'result' in response:
                    return response['result'].get('content')
            except:
                pass
        elif tool_name == 'delete_file':
            # Get file content before deletion
            try:
                response = await self.mcp_server.handle_request({
                    'method': 'tool/read_file',
                    'params': {'file_path': params['file_path']}
                })
                if 'result' in response:
                    return response['result'].get('content')
            except:
                pass
        return None
    
    def _resolve_params(self, params: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Resolve dynamic parameters."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                # Dynamic reference to previous result
                ref = value[1:-1]
                if ref == 'previous_result' and context.results:
                    last_result = context.results[-1]
                    if last_result.success and last_result.data:
                        if isinstance(last_result.data, dict):
                            resolved[key] = last_result.data.get('path', last_result.data)
                        else:
                            resolved[key] = last_result.data
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
    
    async def _wait_before_retry(self, retry_count: int):
        """Wait before retry with exponential backoff."""
        if self.config.exponential_backoff:
            delay = self.config.retry_delay * (2 ** (retry_count - 1))
        else:
            delay = self.config.retry_delay
        
        await asyncio.sleep(min(delay, 10.0))  # Cap at 10 seconds
    
    def _validate_tool_plan(self, tool_plan) -> bool:
        """Validate tool plan."""
        if not tool_plan or not tool_plan.tools:
            return False
        
        # Check for circular dependencies
        for i, tool in enumerate(tool_plan.tools):
            if 'depends_on' in tool:
                if tool['depends_on'] >= i:
                    logger.warning(f"Invalid dependency: tool {i} depends on {tool['depends_on']}")
                    return False
        
        return True
    
    def _update_statistics(self, results: List[ToolResult]):
        """Update execution statistics."""
        for result in results:
            stats = self.execution_stats[result.tool_name]
            stats['total'] += 1
            
            if result.success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            if result.status == ToolStatus.RECOVERED:
                stats['recovered'] += 1
            
            # Update average time
            if result.execution_time > 0:
                current_avg = stats['average_time']
                total = stats['total']
                stats['average_time'] = (current_avg * (total - 1) + result.execution_time) / total
        
        # Report to metrics collector
        if self.metrics:
            for result in results:
                self.metrics.record_request(
                    result.tool_name,
                    result.execution_time,
                    result.success
                )
    
    async def rollback_all(self, context: ExecutionContext) -> bool:
        """Rollback all operations in context."""
        if not self.config.enable_rollback:
            return False
        
        success = True
        for operation in reversed(context.rollback_stack):
            if not await self.rollback.rollback(operation, self.mcp_server):
                logger.error(f"Failed to rollback {operation['tool_name']}")
                success = False
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return dict(self.execution_stats)