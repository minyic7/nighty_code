# src/copilot/guardrails/checker.py
"""
Guardrails for safe and responsible copilot behavior
"""

import logging
from typing import List, Optional

from ..core.types import CopilotState, GuardrailCheck, ToolCall

logger = logging.getLogger(__name__)


class GuardrailChecker:
    """Checks and enforces guardrails for copilot actions"""
    
    def __init__(self):
        self._initialized = False
        self.safety_patterns = [
            "rm -rf /",
            "format c:",
            "delete system32",
            "drop database",
            "truncate table"
        ]
        
        self.sensitive_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "~/.ssh/",
            ".env",
            "credentials",
            "secrets",
            "private_key"
        ]
    
    async def initialize(self):
        """Initialize guardrail checker"""
        if self._initialized:
            return
        
        self._initialized = True
        logger.info("GuardrailChecker initialized")
    
    async def check_all(self, state: CopilotState) -> List[GuardrailCheck]:
        """Run all guardrail checks on the current state"""
        checks = []
        
        # Check for dangerous commands
        checks.append(await self.check_dangerous_commands(state))
        
        # Check for sensitive file access
        checks.append(await self.check_sensitive_files(state))
        
        # Check for resource limits
        checks.append(await self.check_resource_limits(state))
        
        # Check for ethical concerns
        checks.append(await self.check_ethical_guidelines(state))
        
        return [c for c in checks if c is not None]
    
    async def check_dangerous_commands(self, state: CopilotState) -> Optional[GuardrailCheck]:
        """Check for dangerous commands in tool calls"""
        recent_tools = state.tool_history[-5:] if state.tool_history else []
        
        for tool_call in recent_tools:
            if tool_call.status == "pending":
                # Check arguments for dangerous patterns
                args_str = str(tool_call.arguments).lower()
                
                for pattern in self.safety_patterns:
                    if pattern.lower() in args_str:
                        return GuardrailCheck(
                            passed=False,
                            check_type="dangerous_command",
                            message=f"Blocked dangerous command pattern: {pattern}",
                            severity="error",
                            suggestions=["Use safer alternatives", "Request specific file operations"]
                        )
        
        return GuardrailCheck(
            passed=True,
            check_type="dangerous_command",
            message="No dangerous commands detected",
            severity="info"
        )
    
    async def check_sensitive_files(self, state: CopilotState) -> Optional[GuardrailCheck]:
        """Check for access to sensitive files"""
        recent_tools = state.tool_history[-5:] if state.tool_history else []
        
        for tool_call in recent_tools:
            if "file" in tool_call.tool_name.lower() or "read" in tool_call.tool_name.lower():
                args_str = str(tool_call.arguments).lower()
                
                for sensitive_path in self.sensitive_paths:
                    if sensitive_path.lower() in args_str:
                        return GuardrailCheck(
                            passed=False,
                            check_type="sensitive_file_access",
                            message=f"Blocked access to sensitive file: {sensitive_path}",
                            severity="warning",
                            suggestions=["Request non-sensitive files", "Ask user for permission"]
                        )
        
        return GuardrailCheck(
            passed=True,
            check_type="sensitive_file_access",
            message="No sensitive file access detected",
            severity="info"
        )
    
    async def check_resource_limits(self, state: CopilotState) -> Optional[GuardrailCheck]:
        """Check for resource usage limits"""
        # Check number of tool calls
        if len(state.tool_history) > 100:
            return GuardrailCheck(
                passed=False,
                check_type="resource_limit",
                message="Exceeded maximum tool calls limit",
                severity="warning",
                suggestions=["Optimize tool usage", "Break down into smaller tasks"]
            )
        
        # Check memory usage
        total_memories = len(state.short_term_memory) + len(state.long_term_memory)
        if total_memories > 1000:
            return GuardrailCheck(
                passed=False,
                check_type="resource_limit",
                message="Memory limit approaching",
                severity="warning",
                suggestions=["Consolidate memories", "Clear old memories"]
            )
        
        return GuardrailCheck(
            passed=True,
            check_type="resource_limit",
            message="Resource usage within limits",
            severity="info"
        )
    
    async def check_ethical_guidelines(self, state: CopilotState) -> Optional[GuardrailCheck]:
        """Check for ethical concerns in user requests"""
        if not state.messages:
            return None
        
        # Get last user message
        last_user_msg = None
        for msg in reversed(state.messages):
            if not hasattr(msg, 'thoughts'):
                last_user_msg = msg
                break
        
        if not last_user_msg:
            return None
        
        # Check for harmful content patterns
        harmful_patterns = [
            "hack",
            "crack",
            "exploit",
            "malware",
            "virus",
            "ddos",
            "phishing"
        ]
        
        content_lower = last_user_msg.content.lower()
        for pattern in harmful_patterns:
            if pattern in content_lower:
                # Context-aware check - some uses might be legitimate
                if any(safe_word in content_lower for safe_word in ["security", "protect", "defend", "prevent"]):
                    # Likely discussing security defensively
                    continue
                
                return GuardrailCheck(
                    passed=False,
                    check_type="ethical_concern",
                    message=f"Potential harmful request detected: {pattern}",
                    severity="warning",
                    suggestions=["Clarify legitimate use case", "Focus on defensive security"]
                )
        
        return GuardrailCheck(
            passed=True,
            check_type="ethical_concern",
            message="No ethical concerns detected",
            severity="info"
        )
    
    async def validate_tool_call(self, tool_call: ToolCall) -> GuardrailCheck:
        """Validate a specific tool call before execution"""
        # Check tool name
        if tool_call.tool_name in ["delete_file", "remove_directory"]:
            # Extra caution for destructive operations
            path = tool_call.arguments.get("path", "")
            
            # Never allow deletion of system directories
            system_paths = ["/", "/usr", "/bin", "/etc", "/sys", "/var", "C:\\Windows"]
            for sys_path in system_paths:
                if path.startswith(sys_path):
                    return GuardrailCheck(
                        passed=False,
                        check_type="destructive_operation",
                        message=f"Blocked deletion of system path: {path}",
                        severity="error",
                        suggestions=["Target user files only", "Request specific file deletion"]
                    )
        
        return GuardrailCheck(
            passed=True,
            check_type="tool_validation",
            message=f"Tool {tool_call.tool_name} validated",
            severity="info"
        )