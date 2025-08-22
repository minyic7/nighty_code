"""
Adaptive tool chain for progressive tool execution with observability.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from collections import deque
import json


class ExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RECOVERED = "recovered"
    SKIPPED = "skipped"


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    status: ExecutionStatus
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0


@dataclass
class ExecutionStep:
    """Single step in the execution chain."""
    tool_name: str
    params: Dict[str, Any]
    description: str
    timeout: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 3
    fallback_tool: Optional[str] = None
    
    
@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    steps: List[ExecutionStep]
    parallel_groups: List[List[int]] = field(default_factory=list)
    conditional_steps: Dict[int, Callable[[List[ToolResult]], bool]] = field(default_factory=dict)
    

class ProgressCallback:
    """Callback for reporting execution progress."""
    
    def on_start(self, plan: ExecutionPlan):
        """Called when execution starts."""
        pass
    
    def on_step_start(self, step: ExecutionStep, step_index: int):
        """Called when a step starts."""
        pass
    
    def on_step_complete(self, step: ExecutionStep, result: ToolResult, step_index: int):
        """Called when a step completes."""
        pass
    
    def on_retry(self, step: ExecutionStep, attempt: int, error: str):
        """Called when retrying a failed step."""
        pass
    
    def on_fallback(self, original: ExecutionStep, fallback: ExecutionStep):
        """Called when using fallback tool."""
        pass
    
    def on_complete(self, results: List[ToolResult]):
        """Called when execution completes."""
        pass


class ConsoleProgressCallback(ProgressCallback):
    """Console output progress callback."""
    
    def on_start(self, plan: ExecutionPlan):
        print(f"Starting execution with {len(plan.steps)} steps...")
    
    def on_step_start(self, step: ExecutionStep, step_index: int):
        print(f"  [{step_index+1}] {step.description}")
    
    def on_step_complete(self, step: ExecutionStep, result: ToolResult, step_index: int):
        status_symbol = "[OK]" if result.status == ExecutionStatus.SUCCESS else "[X]"
        print(f"    {status_symbol} {result.status.value} ({result.execution_time:.2f}s)")
        if result.error:
            print(f"      Error: {result.error}")
    
    def on_retry(self, step: ExecutionStep, attempt: int, error: str):
        print(f"    Retrying {step.tool_name} (attempt {attempt}): {error}")
    
    def on_fallback(self, original: ExecutionStep, fallback: ExecutionStep):
        print(f"    Falling back from {original.tool_name} to {fallback.tool_name}")
    
    def on_complete(self, results: List[ToolResult]):
        success_count = sum(1 for r in results if r.status in [ExecutionStatus.SUCCESS, ExecutionStatus.RECOVERED])
        print(f"Execution complete: {success_count}/{len(results)} successful")


class AdaptiveToolChain:
    """
    Manages progressive tool execution with recovery strategies.
    Provides observable execution for transparency.
    """
    
    def __init__(self, mcp_client, progress_callback: Optional[ProgressCallback] = None):
        """
        Initialize tool chain.
        
        Args:
            mcp_client: MCP client for tool execution
            progress_callback: Optional callback for progress reporting
        """
        self.mcp_client = mcp_client
        self.progress_callback = progress_callback or ConsoleProgressCallback()
        self.execution_history = deque(maxlen=100)
        self.recovery_strategies = self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different failure types."""
        return {
            "file_not_found": self._recover_file_not_found,
            "timeout": self._recover_timeout,
            "permission_denied": self._recover_permission_denied,
            "network_error": self._recover_network_error,
            "generic": self._recover_generic,
        }
    
    async def execute(self, plan: ExecutionPlan) -> List[ToolResult]:
        """
        Execute a tool chain plan.
        
        Args:
            plan: Execution plan with steps
            
        Returns:
            List of tool results
        """
        results = []
        self.progress_callback.on_start(plan)
        
        # Execute steps in order, respecting parallel groups
        step_index = 0
        while step_index < len(plan.steps):
            # Check for parallel group
            parallel_group = None
            for group in plan.parallel_groups:
                if step_index in group:
                    parallel_group = group
                    break
            
            if parallel_group:
                # Execute parallel steps
                group_results = await self._execute_parallel(
                    [plan.steps[i] for i in parallel_group],
                    parallel_group
                )
                results.extend(group_results)
                step_index = max(parallel_group) + 1
            else:
                # Check conditional execution
                if step_index in plan.conditional_steps:
                    condition = plan.conditional_steps[step_index]
                    if not condition(results):
                        results.append(ToolResult(
                            tool_name=plan.steps[step_index].tool_name,
                            status=ExecutionStatus.SKIPPED,
                            data=None,
                            error="Condition not met"
                        ))
                        step_index += 1
                        continue
                
                # Execute single step
                result = await self._execute_step(plan.steps[step_index], step_index)
                results.append(result)
                step_index += 1
        
        self.progress_callback.on_complete(results)
        self.execution_history.append({
            'timestamp': time.time(),
            'plan': plan,
            'results': results
        })
        
        return results
    
    async def _execute_step(self, step: ExecutionStep, step_index: int) -> ToolResult:
        """Execute a single step with retry and fallback logic."""
        self.progress_callback.on_step_start(step, step_index)
        
        attempt = 0
        last_error = None
        
        while attempt <= step.max_retries:
            try:
                start_time = time.time()
                
                # Execute tool with timeout
                result_data = await asyncio.wait_for(
                    self._call_tool(step.tool_name, step.params),
                    timeout=step.timeout
                )
                
                result = ToolResult(
                    tool_name=step.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    data=result_data,
                    execution_time=time.time() - start_time,
                    retry_count=attempt
                )
                
                self.progress_callback.on_step_complete(step, result, step_index)
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout}s"
                if attempt < step.max_retries and step.retry_on_failure:
                    self.progress_callback.on_retry(step, attempt + 1, last_error)
                    attempt += 1
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    break
                    
            except Exception as e:
                last_error = str(e)
                if attempt < step.max_retries and step.retry_on_failure:
                    self.progress_callback.on_retry(step, attempt + 1, last_error)
                    attempt += 1
                    await asyncio.sleep(1)
                else:
                    break
        
        # All retries failed - try recovery
        if step.fallback_tool:
            # Transform params for fallback tool
            fallback_params = self._transform_params_for_fallback(
                step.tool_name, step.fallback_tool, step.params
            )
            
            fallback_step = ExecutionStep(
                tool_name=step.fallback_tool,
                params=fallback_params,
                description=f"Fallback: {step.description}",
                timeout=step.timeout
            )
            self.progress_callback.on_fallback(step, fallback_step)
            
            try:
                start_time = time.time()
                result_data = await asyncio.wait_for(
                    self._call_tool(fallback_step.tool_name, fallback_step.params),
                    timeout=fallback_step.timeout
                )
                
                result = ToolResult(
                    tool_name=fallback_step.tool_name,
                    status=ExecutionStatus.RECOVERED,
                    data=result_data,
                    execution_time=time.time() - start_time,
                    retry_count=attempt,
                    error=f"Original tool failed: {last_error}"
                )
                
                self.progress_callback.on_step_complete(step, result, step_index)
                return result
                
            except Exception as e:
                last_error = f"Fallback also failed: {e}"
        
        # Complete failure
        result = ToolResult(
            tool_name=step.tool_name,
            status=ExecutionStatus.FAILED,
            data=None,
            error=last_error,
            retry_count=attempt
        )
        
        self.progress_callback.on_step_complete(step, result, step_index)
        return result
    
    async def _execute_parallel(self, steps: List[ExecutionStep], indices: List[int]) -> List[ToolResult]:
        """Execute multiple steps in parallel."""
        tasks = [
            self._execute_step(step, idx)
            for step, idx in zip(steps, indices)
        ]
        return await asyncio.gather(*tasks)
    
    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call MCP tool."""
        # Call MCP server's handle_request method with correct format
        response = await self.mcp_client.handle_request({
            "jsonrpc": "2.0",
            "method": f"tool/{tool_name}",
            "params": params
        })
        
        if "error" in response:
            raise Exception(response["error"])
        
        return response.get("result")
    
    def _recover_file_not_found(self, step: ExecutionStep, error: str) -> Optional[ExecutionStep]:
        """Recovery strategy for file not found errors."""
        # Try fuzzy finding
        if 'file_path' in step.params:
            # Extract just the filename from path
            filename = step.params['file_path'].split('/')[-1].split('\\')[-1]
            return ExecutionStep(
                tool_name="fuzzy_find",
                params={"query": filename},
                description=f"Fuzzy find: {filename}",
                timeout=step.timeout
            )
        return None
    
    def _recover_timeout(self, step: ExecutionStep, error: str) -> Optional[ExecutionStep]:
        """Recovery strategy for timeout errors."""
        # Try with increased timeout
        new_step = ExecutionStep(
            tool_name=step.tool_name,
            params=step.params,
            description=step.description,
            timeout=step.timeout * 2,
            max_retries=1
        )
        return new_step
    
    def _recover_permission_denied(self, step: ExecutionStep, error: str) -> Optional[ExecutionStep]:
        """Recovery strategy for permission errors."""
        # Try alternative tool
        if step.tool_name == "read_file":
            return ExecutionStep(
                tool_name="list_directory",
                params={"path": str(step.params.get('file_path', '.'))},
                description="List directory instead of reading file",
                timeout=step.timeout
            )
        return None
    
    def _recover_network_error(self, step: ExecutionStep, error: str) -> Optional[ExecutionStep]:
        """Recovery strategy for network errors."""
        # Retry with exponential backoff
        return ExecutionStep(
            tool_name=step.tool_name,
            params=step.params,
            description=step.description,
            timeout=step.timeout * 1.5,
            max_retries=2
        )
    
    def _recover_generic(self, step: ExecutionStep, error: str) -> Optional[ExecutionStep]:
        """Generic recovery strategy."""
        # Try smart suggest for alternatives
        return ExecutionStep(
            tool_name="smart_suggest",
            params={"context": {"failed_tool": step.tool_name, "error": error}},
            description="Get suggestions after failure",
            timeout=10.0
        )
    
    def create_plan_from_intent(self, intent, context: Dict[str, Any]) -> ExecutionPlan:
        """
        Create execution plan from recognized intent.
        
        Args:
            intent: Recognized intent from IntentRecognizer
            context: Current context
            
        Returns:
            Execution plan with steps
        """
        steps = []
        
        # Build steps based on intent type and suggested tools
        for tool_name in intent.suggested_tools:
            params = self._build_tool_params(tool_name, intent.entities, context)
            
            step = ExecutionStep(
                tool_name=tool_name,
                params=params,
                description=self._describe_tool_action(tool_name, params),
                timeout=30.0,
                retry_on_failure=True,
                max_retries=2,
                fallback_tool=self._get_fallback_tool(tool_name)
            )
            steps.append(step)
        
        # Identify parallel opportunities
        parallel_groups = self._identify_parallel_groups(steps)
        
        # Add conditional steps if needed
        conditional_steps = {}
        if intent.confidence < 0.7:
            # Low confidence - add exploration step
            conditional_steps[len(steps)] = lambda results: any(
                r.status == ExecutionStatus.FAILED for r in results
            )
            steps.append(ExecutionStep(
                tool_name="smart_suggest",
                params={"context": context},
                description="Get suggestions for better results",
                timeout=10.0
            ))
        
        return ExecutionPlan(
            steps=steps,
            parallel_groups=parallel_groups,
            conditional_steps=conditional_steps
        )
    
    def _build_tool_params(self, tool_name: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """Build parameters for tool based on entities and context."""
        params = {}
        
        if tool_name == "read_file":
            if 'file_paths' in entities:
                params['file_path'] = entities['file_paths'][0]
            elif 'current_file' in context:
                params['file_path'] = context['current_file']
                
        elif tool_name == "list_directory":
            if 'directories' in entities:
                params['directory_path'] = entities['directories'][0]
            elif 'current_directory' in context:
                params['directory_path'] = context['current_directory']
            else:
                params['directory_path'] = '.'
                
        elif tool_name == "search_files":
            if 'quoted_terms' in entities:
                params['query'] = entities['quoted_terms'][0]
            elif 'code_elements' in entities:
                params['query'] = entities['code_elements'][0]
            if 'extensions' in entities:
                params['extensions'] = entities['extensions']
                
        elif tool_name == "search_pattern":
            if 'quoted_terms' in entities:
                params['pattern'] = entities['quoted_terms'][0]
            if 'file_paths' in entities:
                params['paths'] = entities['file_paths']
                
        elif tool_name == "fuzzy_find":
            if 'file_paths' in entities:
                params['query'] = entities['file_paths'][0]
            elif 'quoted_terms' in entities:
                params['query'] = entities['quoted_terms'][0]
                
        elif tool_name == "smart_suggest":
            params['context'] = context
        
        return params
    
    def _describe_tool_action(self, tool_name: str, params: Dict) -> str:
        """Generate human-readable description of tool action."""
        descriptions = {
            "read_file": f"Reading file: {params.get('file_path', 'unknown')}",
            "list_directory": f"Listing directory: {params.get('directory_path', '.')}",
            "search_files": f"Searching for: {params.get('query', 'files')}",
            "search_pattern": f"Searching pattern: {params.get('pattern', 'unknown')}",
            "fuzzy_find": f"Fuzzy finding: {params.get('query', 'unknown')}",
            "smart_suggest": "Getting smart suggestions",
        }
        return descriptions.get(tool_name, f"Executing {tool_name}")
    
    def _get_fallback_tool(self, tool_name: str) -> Optional[str]:
        """Get fallback tool for a given tool."""
        fallbacks = {
            "read_file": "fuzzy_find",
            "search_files": "fuzzy_find",
            "search_pattern": "search_files",
        }
        return fallbacks.get(tool_name)
    
    def _identify_parallel_groups(self, steps: List[ExecutionStep]) -> List[List[int]]:
        """Identify steps that can be executed in parallel."""
        # Simple heuristic: same tool type can run in parallel
        groups = []
        used_indices = set()
        
        for i, step in enumerate(steps):
            if i in used_indices:
                continue
                
            group = [i]
            for j in range(i + 1, len(steps)):
                if j not in used_indices and step.tool_name == steps[j].tool_name:
                    # Same tool, can parallelize
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) > 1:
                groups.append(group)
                used_indices.add(i)
        
        return groups