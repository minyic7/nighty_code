# src/copilot/tools/router.py
"""
MCP Tool Router - Routes tool calls to appropriate MCP servers
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.mcp import MCPManager, BaseMCPServer
from src.llm import LLMClient, Message, MessageRole
from ..core.types import (
    CopilotState,
    ToolCall,
    ToolStatus,
    ActionPlan
)

logger = logging.getLogger(__name__)


class ToolSelectionOutput(BaseModel):
    """Structured output for tool selection"""
    tool_name: str = Field(description="Name of the tool to use")
    server_name: str = Field(description="Name of the MCP server that provides this tool")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")
    reasoning: str = Field(description="Why this tool was selected")


class ToolExecutionPlan(BaseModel):
    """Plan for executing multiple tools"""
    tools: List[ToolSelectionOutput] = Field(description="Tools to execute in order")
    can_parallelize: List[bool] = Field(description="Whether each tool can run in parallel with next")
    expected_outcomes: List[str] = Field(description="Expected outcome from each tool")


class MCPToolRouter:
    """Routes and executes tool calls through MCP servers"""
    
    def __init__(self, mcp_manager: MCPManager, llm_client: Optional[LLMClient] = None):
        self.mcp_manager = mcp_manager
        self.llm_client = llm_client
        self._tool_catalog: Dict[str, Tuple[str, BaseMCPServer]] = {}  # tool_name -> (server_name, server)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the router and discover available tools"""
        if not self.llm_client:
            from src.llm import get_llm_manager, LLMProvider
            manager = await get_llm_manager()
            self.llm_client = manager.get_client(LLMProvider.ANTHROPIC)
        
        # Discover tools from all MCP servers
        await self._discover_tools()
        self._initialized = True
    
    async def _discover_tools(self):
        """Discover all available tools from MCP servers"""
        self._tool_catalog.clear()
        
        servers = self.mcp_manager.get_servers()
        for server_name, server in servers.items():
            try:
                # Get tools from server
                tools = server.list_tools()
                for tool in tools:
                    # ToolDefinition is a dataclass, not a dict
                    tool_name = tool.name if hasattr(tool, 'name') else tool.get("name", "")
                    if tool_name:
                        self._tool_catalog[tool_name] = (server_name, server)
                        logger.info(f"Discovered tool '{tool_name}' from server '{server_name}'")
            except Exception as e:
                logger.error(f"Error discovering tools from {server_name}: {e}")
    
    async def select_tools(self, state: CopilotState, task: str) -> List[ToolSelectionOutput]:
        """Select appropriate tools for a task using LLM"""
        if not self._initialized:
            await self.initialize()
        
        # Format available tools
        tools_description = self._format_tool_catalog()
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a tool selector. Choose the right tools for the task and provide correct arguments."
            ),
            Message(
                MessageRole.USER,
                f"Task: {task}\n\n"
                f"Available tools:\n{tools_description}\n\n"
                "Select tools and arguments to accomplish this task."
            )
        ]
        
        # Get tool execution plan
        plan = await self.llm_client.complete(
            messages,
            response_model=ToolExecutionPlan,
            temperature=0.3
        )
        
        return plan.tools
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolCall:
        """Execute a single tool call through MCP"""
        if not self._initialized:
            await self.initialize()
        
        tool_call.status = ToolStatus.RUNNING
        
        try:
            # Find the server for this tool
            if tool_call.tool_name not in self._tool_catalog:
                raise ValueError(f"Unknown tool: {tool_call.tool_name}")
            
            server_name, server = self._tool_catalog[tool_call.tool_name]
            
            # Verify server name matches if specified
            if tool_call.server_name and tool_call.server_name != server_name:
                raise ValueError(
                    f"Tool {tool_call.tool_name} is provided by {server_name}, "
                    f"not {tool_call.server_name}"
                )
            
            # Execute the tool - convert to MCP ToolCall format
            from src.mcp import ToolCall as MCPToolCall
            
            # Normalize arguments based on tool name
            normalized_args = self._normalize_tool_arguments(
                tool_call.tool_name, 
                tool_call.arguments
            )
            
            mcp_tool_call = MCPToolCall(
                name=tool_call.tool_name,
                arguments=normalized_args
            )
            result = await server.call_tool(mcp_tool_call)
            
            tool_call.result = result
            tool_call.status = ToolStatus.SUCCESS
            logger.info(f"Successfully executed tool: {tool_call.tool_name}")
            
        except Exception as e:
            tool_call.error = str(e)
            tool_call.status = ToolStatus.FAILED
            logger.error(f"Failed to execute tool {tool_call.tool_name}: {e}")
        
        return tool_call
    
    async def execute_plan_step(self, state: CopilotState) -> CopilotState:
        """Execute the next step in the current plan"""
        if not state.current_plan or state.current_plan.status != "executing":
            return state
        
        plan = state.current_plan
        
        # Check if we've completed all steps
        if plan.current_step >= len(plan.steps):
            plan.status = "completed"
            return state
        
        # Get current step
        current_step = plan.steps[plan.current_step]
        
        # Select tools for this step
        tool_selections = await self.select_tools(state, current_step)
        
        # Execute selected tools
        for selection in tool_selections:
            tool_call = ToolCall(
                tool_name=selection.tool_name,
                server_name=selection.server_name,
                arguments=selection.arguments
            )
            
            # Execute the tool
            executed_call = await self.execute_tool(tool_call)
            
            # Record in state
            state.record_tool_call(executed_call)
            
            # Check if tool failed and we should stop
            if executed_call.status == ToolStatus.FAILED:
                logger.warning(f"Tool {executed_call.tool_name} failed")
                # Stop execution on critical failure
                plan.status = "failed"
                return state
        
        # Move to next step
        plan.current_step += 1
        
        return state
    
    async def route_tool_request(self, state: CopilotState, request: str) -> Tuple[str, Any]:
        """Route a natural language tool request to the appropriate tool"""
        if not self._initialized:
            await self.initialize()
        
        # Use LLM to understand the request and select tool
        tools_description = self._format_tool_catalog()
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are a tool router. Parse the user request and select the appropriate tool."
            ),
            Message(
                MessageRole.USER,
                f"Request: {request}\n\n"
                f"Available tools:\n{tools_description}\n\n"
                "Select the best tool and provide arguments."
            )
        ]
        
        selection = await self.llm_client.complete(
            messages,
            response_model=ToolSelectionOutput,
            temperature=0.2
        )
        
        # Create and execute tool call
        tool_call = ToolCall(
            tool_name=selection.tool_name,
            server_name=selection.server_name,
            arguments=selection.arguments
        )
        
        executed_call = await self.execute_tool(tool_call)
        state.record_tool_call(executed_call)
        
        return selection.tool_name, executed_call.result
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get a mapping of server names to tool names"""
        tools_by_server: Dict[str, List[str]] = {}
        
        for tool_name, (server_name, _) in self._tool_catalog.items():
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            tools_by_server[server_name].append(tool_name)
        
        return tools_by_server
    
    def _format_tool_catalog(self) -> str:
        """Format the tool catalog for LLM context"""
        if not self._tool_catalog:
            return "No tools available"
        
        formatted = []
        tools_by_server = self.get_available_tools()
        
        for server_name, tool_names in tools_by_server.items():
            formatted.append(f"\n{server_name} server:")
            for tool_name in tool_names:
                # Could add tool descriptions here if available
                formatted.append(f"  - {tool_name}")
        
        return "\n".join(formatted)
    
    async def validate_tool_arguments(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate tool arguments before execution"""
        if tool_name not in self._tool_catalog:
            return False, f"Unknown tool: {tool_name}"
        
        server_name, server = self._tool_catalog[tool_name]
        
        try:
            # Get tool schema from server if available
            tools = server.list_tools()
            tool_schema = None
            
            for tool in tools:
                tool_name_in_list = tool.name if hasattr(tool, 'name') else tool.get("name", "")
                if tool_name_in_list == tool_name:
                    tool_schema = tool
                    break
            
            if not tool_schema:
                return False, f"Tool {tool_name} not found in server {server_name}"
            
            # Basic validation - check required parameters
            # This could be enhanced with more sophisticated schema validation
            required_params = tool_schema.get("required_params", [])
            for param in required_params:
                if param not in arguments:
                    return False, f"Missing required parameter: {param}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating arguments: {str(e)}"
    
    def _normalize_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool arguments to match expected parameter names"""
        normalized = arguments.copy()
        
        # Common parameter mappings
        if tool_name == "read_file":
            # Map file_path -> path
            if "file_path" in normalized and "path" not in normalized:
                normalized["path"] = normalized.pop("file_path")
        
        elif tool_name == "search_pattern":
            # Map directory -> path
            if "directory" in normalized and "path" not in normalized:
                normalized["path"] = normalized.pop("directory")
        
        elif tool_name == "list_directory":
            # Ensure path is relative if it starts with /
            if "path" in normalized:
                path = normalized["path"]
                # Remove leading / to make it relative
                if path.startswith("/"):
                    normalized["path"] = path.lstrip("/")
        
        # For all tools, clean up the path
        if "path" in normalized:
            path = normalized["path"]
            # Convert absolute paths to relative
            if path.startswith("/Users/minyic/project/nighty_code/"):
                normalized["path"] = path.replace("/Users/minyic/project/nighty_code/", "")
            elif path.startswith("/"):
                normalized["path"] = path.lstrip("/")
        
        return normalized