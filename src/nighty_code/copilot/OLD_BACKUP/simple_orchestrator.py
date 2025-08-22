"""
Simple orchestrator that coordinates components.
Clear separation of concerns.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .simple_intent import SimpleIntentRecognizer, IntentType
from .simple_validator import SimpleInputValidator
from .simple_tool_executor import SimpleToolExecutor, ToolResult

logger = logging.getLogger(__name__)


class SimpleCopilotOrchestrator:
    """
    Simple orchestrator that coordinates everything.
    Each component has a single responsibility.
    """
    
    def __init__(
        self,
        mcp_server,
        project_path: Optional[Path] = None,
        enable_validation: bool = True,
        tool_timeout: float = 5.0
    ):
        self.mcp_server = mcp_server
        self.project_path = project_path or Path.cwd()
        
        # Initialize components
        self.recognizer = SimpleIntentRecognizer()
        self.validator = SimpleInputValidator() if enable_validation else None
        self.executor = SimpleToolExecutor(mcp_server, default_timeout=tool_timeout)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        Simple, clear flow.
        """
        # 1. Validate input (if enabled)
        if self.validator:
            try:
                query = self.validator.validate_query(query)
            except ValueError as e:
                return {
                    'success': False,
                    'error': f"Invalid query: {e}",
                    'results': []
                }
        
        # 2. Recognize intent
        intent = self.recognizer.recognize(query)
        
        # 3. Build tool plan
        tool_plan = self._build_tool_plan(intent)
        
        # 4. Execute tools
        if tool_plan:
            results = await self.executor.execute_multiple(tool_plan)
        else:
            results = []
        
        # 5. Return results
        return {
            'success': True,
            'intent': intent.type.value,
            'tools_suggested': intent.suggested_tools,
            'tools_executed': [t['name'] for t in tool_plan],
            'results': results
        }
    
    def _build_tool_plan(self, intent) -> List[Dict[str, Any]]:
        """
        Build execution plan from intent.
        Simple mapping, no magic.
        """
        plan = []
        
        # Map intent to concrete tool calls
        if intent.type == IntentType.FILE_READ:
            # Find the file entity
            file_entities = [e for e in intent.entities if e.type == 'file']
            if file_entities:
                plan.append({
                    'name': 'read_file',
                    'params': {'file_path': file_entities[0].value}
                })
            elif intent.entities:
                # Try fuzzy find first
                plan.append({
                    'name': 'fuzzy_find',
                    'params': {'query': intent.entities[0].value}
                })
        
        elif intent.type == IntentType.DIR_LIST:
            dir_entities = [e for e in intent.entities if e.type == 'dir']
            plan.append({
                'name': 'list_directory',
                'params': {'directory_path': dir_entities[0].value if dir_entities else '.'}
            })
        
        elif intent.type == IntentType.CODE_SEARCH:
            pattern_entities = [e for e in intent.entities if e.type in ['pattern', 'literal']]
            if pattern_entities:
                plan.append({
                    'name': 'search_pattern',
                    'params': {'pattern': pattern_entities[0].value}
                })
        
        elif intent.type == IntentType.FILE_SEARCH:
            # Use first entity as search query
            if intent.entities:
                plan.append({
                    'name': 'search_files',
                    'params': {'query': intent.entities[0].value}
                })
        
        return plan
    
    def validate_path(self, path: str) -> bool:
        """
        Validate a path using the validator.
        Separate method for explicit path validation.
        """
        if not self.validator:
            return True
        
        try:
            self.validator.validate_path(path)
            return True
        except ValueError:
            return False