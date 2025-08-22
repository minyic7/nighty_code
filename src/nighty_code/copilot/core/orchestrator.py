"""
Query orchestration - coordinates intent recognition, validation, and tool execution.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .intent import IntentRecognizer, IntentType
from .validator import InputValidator
from .tools import ToolExecutor, ExecutionStatus

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """
    Orchestrates the complete query processing pipeline.
    Coordinates intent recognition, validation, and tool execution.
    """
    
    def __init__(
        self,
        mcp_server=None,
        llm_client=None,
        project_path: Optional[Path] = None,
        enable_validation: bool = True,
        enable_tools: bool = True,
        tool_timeout: float = 5.0
    ):
        """
        Initialize query orchestrator.
        
        Args:
            mcp_server: Optional MCP server for tool execution
            llm_client: Optional LLM client for intent recognition
            project_path: Project root path
            enable_validation: Whether to validate inputs
            enable_tools: Whether to execute tools
            tool_timeout: Timeout for tool execution
        """
        self.project_path = project_path or Path.cwd()
        self.enable_validation = enable_validation
        self.enable_tools = enable_tools and mcp_server is not None
        
        # Initialize components
        self.intent_recognizer = IntentRecognizer(
            llm_client=llm_client,
            project_root=self.project_path,
            enable_llm_fallback=llm_client is not None
        )
        
        if enable_validation:
            self.validator = InputValidator()
        else:
            self.validator = None
        
        if self.enable_tools:
            self.tool_executor = ToolExecutor(
                mcp_server=mcp_server,
                default_timeout=tool_timeout
            )
        else:
            self.tool_executor = None
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_chaining: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline.
        
        Args:
            query: User's query
            context: Optional execution context
            
        Returns:
            Dictionary with:
                - success: Whether processing succeeded
                - intent: Recognized intent type
                - confidence: Intent confidence
                - tools_executed: List of executed tools
                - results: Tool execution results
                - error: Error message if failed
        """
        # 1. Validate input
        if self.validator:
            try:
                query = self.validator.validate_query(query)
            except ValueError as e:
                return {
                    'success': False,
                    'error': f"Invalid query: {e}",
                    'intent': None,
                    'results': []
                }
        
        # 2. Recognize intent
        try:
            intent = await self.intent_recognizer.recognize(
                query,
                context or {'project_path': str(self.project_path)}
            )
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            return {
                'success': False,
                'error': f"Intent recognition failed: {e}",
                'intent': None,
                'results': []
            }
        
        # 3. Build response structure
        response = {
            'success': True,
            'intent': intent.type.value,
            'confidence': intent.confidence,
            'reasoning': intent.reasoning,
            'ambiguous': intent.ambiguous,
            'entities': [(e.type.value, e.value) for e in intent.entities],
            'suggested_tools': intent.suggested_tools,
            'tools_executed': [],
            'results': []
        }
        
        # 4. Execute tools if enabled - let LLM decide what tools to use
        if self.enable_tools and self.tool_executor:
            try:
                # Check if we should use chained execution for complex queries
                if use_chaining and self._should_use_chaining(query, intent):
                    logger.info("Using chained tool execution for complex query")
                    # Use chained execution for better results
                    chained_results = await self._execute_chained_tools(
                        query, intent, context or {}
                    )
                    
                    # Process chained results
                    for step_result in chained_results:
                        response['tools_executed'].extend(step_result.get('tools_executed', []))
                        response['results'].extend(step_result.get('results', []))
                    
                    response['execution_type'] = 'chained'
                    response['chain_steps'] = len(chained_results)
                    
                else:
                    # Use parallel execution for simple queries
                    tools_to_execute = await self._llm_select_tools(query, intent, context or {})
                    
                    if tools_to_execute:
                        # Execute the LLM-selected tools in parallel for better performance
                        results = await asyncio.wait_for(
                            self.tool_executor.execute_multiple(
                                tools_to_execute,
                                parallel=True  # Enable parallel execution
                            ),
                            timeout=15.0  # Increase timeout for parallel execution
                        )
                        
                        # Process results
                        for result in results:
                            response['tools_executed'].append(result.tool_name)
                            
                            if result.status == ExecutionStatus.SUCCESS:
                                response['results'].append({
                                    'tool': result.tool_name,
                                    'status': 'success',
                                    'data': result.data
                                })
                            else:
                                response['results'].append({
                                    'tool': result.tool_name,
                                    'status': result.status.value,
                                    'error': result.error
                                })
                    else:
                        response['tools_skipped'] = True
                        response['skip_reason'] = "LLM determined no tools needed"
                    
            except asyncio.TimeoutError:
                logger.warning("Tool execution timed out")
                response['tools_skipped'] = True
                response['skip_reason'] = "Tool execution timeout"
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                response['tools_skipped'] = True
                response['skip_reason'] = f"Tool execution error: {e}"
        
        return response
    
    async def _llm_select_tools(self, query: str, intent, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to intelligently select which tools to execute for a query.
        
        Args:
            query: Original user query
            intent: Recognized intent object
            context: Execution context
            
        Returns:
            List of tool specifications to execute
        """
        # Quick keyword-based tool selection for common queries
        query_lower = query.lower()
        import re
        
        # Helper for fuzzy matching
        def fuzzy_match(text, patterns, threshold=0.8):
            """Check if text contains any pattern allowing for small typos."""
            for pattern in patterns:
                if pattern in text:
                    return True
                # Check for common typos (1-2 character difference)
                # Simple approach: check if most of the pattern is in the text
                if len(pattern) > 4:
                    # For longer words, allow some variation
                    pattern_chars = set(pattern)
                    for word in text.split():
                        if len(word) >= len(pattern) - 2:
                            word_chars = set(word.lower())
                            overlap = len(pattern_chars & word_chars) / len(pattern_chars)
                            if overlap >= threshold:
                                return True
            return False
        
        # Multiple tools for comprehensive exploration
        tools_to_execute = []
        
        # 1. Directory exploration - always useful for context
        if any(keyword in query_lower for keyword in ['folder', 'directory', 'files', "what's in", 'show me', 'list', 'structure', 'project', 'explore']):
            logger.info(f"Directory exploration triggered: {query}")
            tools_to_execute.append({
                'name': 'list_directory',
                'params': {'directory_path': '.'},
                'timeout': 5.0
            })
        
        # 2. Module/package exploration - when asking about specific modules
        # Check for common typos in module names
        module_patterns = ['copilot', 'copilt', 'copiolt', 'copliot']  # Common typos
        llm_patterns = ['llm', 'lmm', 'llms']
        mcp_patterns = ['mcp', 'mpc', 'mcps']
        
        # Handle requests to "read code" or "show code" - be proactive!
        if any(keyword in query_lower for keyword in ['read', 'show', 'view', 'see', 'look at', 'check']) and \
           any(keyword in query_lower for keyword in ['code', 'implementation', 'source', 'module', 'package']):
            # User wants to see code - find and read it proactively
            
            if fuzzy_match(query_lower, mcp_patterns):
                # MCP module requested
                # Check if asking specifically about tools
                if 'tools' in query_lower or 'tool' in query_lower:
                    tools_to_execute.append({
                        'name': 'list_directory',
                        'params': {'directory_path': 'src/nighty_code/mcp/tools'},
                        'timeout': 5.0
                    })
                else:
                    tools_to_execute.append({
                        'name': 'list_directory',
                        'params': {'directory_path': 'src/nighty_code/mcp'},
                        'timeout': 5.0
                    })
                # Read main MCP files
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/mcp/__init__.py'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/mcp/server.py'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/mcp/base.py'},
                    'timeout': 5.0
                })
            
            elif fuzzy_match(query_lower, llm_patterns):
                # LLM module requested
                tools_to_execute.append({
                    'name': 'list_directory',
                    'params': {'directory_path': 'src/nighty_code/llm'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/llm/__init__.py'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/llm/client.py'},
                    'timeout': 5.0
                })
            
            elif fuzzy_match(query_lower, module_patterns):
                # Copilot module requested
                tools_to_execute.append({
                    'name': 'list_directory',
                    'params': {'directory_path': 'src/nighty_code/copilot'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/copilot/__init__.py'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/copilot/client.py'},
                    'timeout': 5.0
                })
        
        elif any(keyword in query_lower for keyword in ['module', 'package', 'import', 'from']) or \
           fuzzy_match(query_lower, module_patterns + llm_patterns + mcp_patterns):
            # Look for module names (including typos)
            if fuzzy_match(query_lower, module_patterns):
                tools_to_execute.append({
                    'name': 'list_directory',
                    'params': {'directory_path': 'src/nighty_code/copilot'},
                    'timeout': 5.0
                })
                # Also read the main file
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/copilot/__init__.py'},
                    'timeout': 5.0
                })
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'src/nighty_code/copilot/client.py'},
                    'timeout': 5.0
                })
            
        # Also check for nighty_code typos (common: nighty_cdoe, nightycode, etc.)
        nighty_patterns = ['nighty_code', 'nighty_cdoe', 'nightycode', 'nighty code', 'nitecode']
        if fuzzy_match(query_lower, nighty_patterns):
            # If they mention nighty_code (even with typos), they probably want info about it
            if not tools_to_execute:  # Only add if we haven't already
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': 'README.md'},
                    'timeout': 5.0
                })
        
        # 3. How to use - read documentation and examples
        if any(keyword in query_lower for keyword in ['how to use', 'usage', 'example', 'how do i', 'how can i', 'demonstrate', 'show me how']):
            tools_to_execute.append({
                'name': 'read_file',
                'params': {'file_path': 'README.md'},
                'timeout': 5.0
            })
            # Look for examples
            tools_to_execute.append({
                'name': 'find_files',
                'params': {'pattern': '*example*.py'},
                'timeout': 5.0
            })
        
        # 4. File reading - when specific files are mentioned
        file_pattern = re.findall(r'[\w\-/\\]+\.(?:py|md|txt|json|yaml|yml)', query)
        if file_pattern:
            for file_path in file_pattern:
                tools_to_execute.append({
                    'name': 'read_file',
                    'params': {'file_path': file_path},
                    'timeout': 5.0
                })
        
        # 5. What does X do - search for function/class definitions
        if any(keyword in query_lower for keyword in ['what does', 'what is', 'explain', 'purpose of']):
            # Extract the thing being asked about
            match = re.search(r'what (?:does|is) (\w+)', query_lower)
            if match:
                search_term = match.group(1)
                tools_to_execute.append({
                    'name': 'search_in_files',
                    'params': {'pattern': f'class {search_term}|def {search_term}', 'file_pattern': '*.py'},
                    'timeout': 5.0
                })
        
        # Get access to the LLM client for validation and enhancement
        llm_client = getattr(self, 'main_llm_client', None)
        if not llm_client:
            # Fallback to intent recognizer's LLM client
            llm_client = getattr(self.intent_recognizer, 'llm_client', None)
        
        # If we have keyword matches AND an LLM, validate and enhance
        if tools_to_execute and llm_client:
            logger.info(f"Keyword matching found {len(tools_to_execute)} tools, validating with LLM...")
            # Pass to LLM for validation and potential enhancement
            validated_tools = await self._llm_validate_and_enhance_tools(
                query, tools_to_execute, intent, context, llm_client
            )
            if validated_tools:
                return validated_tools
            # If validation fails, continue with keyword matches
            return tools_to_execute
        
        # If we have keyword matches but no LLM, use them directly
        if tools_to_execute:
            logger.info(f"Using {len(tools_to_execute)} keyword-matched tools without LLM validation")
            return tools_to_execute
        
        # No keyword matches - try pure LLM selection if available
        if not llm_client:
            # No LLM and no keyword matches - fallback
            return self._fallback_tool_selection(intent, context)
        
        # Build prompt for LLM tool selection
        available_tools = [
            "read_file - Read the contents of a specific file (params: file_path)",
            "list_directory - List files and directories in a path (params: directory_path)", 
            "search_in_files - Search for text/patterns in files (params: pattern, file_pattern)",
            "find_files - Find files by name/pattern (params: pattern)",
            "fuzzy_find - Find files with fuzzy matching (params: query)"
        ]
        
        prompt = f"""You are a tool selection assistant. Analyze the user query and select appropriate tools to gather information.

User Query: "{query}"

Intent Recognized: {intent.type.value}
Intent Reasoning: {intent.reasoning}
Detected Entities: {[(e.type.value, e.value) for e in intent.entities]}

Available Tools:
- list_directory: List files and directories (params: directory_path)
- read_file: Read file contents (params: file_path)
- search_in_files: Search text in files (params: pattern, file_pattern)
- find_files: Find files by name (params: pattern)
- fuzzy_find: Fuzzy file search (params: query)

Project Context:
- Current directory: {context.get('current_directory', '.')}
- Project path: {context.get('project_path', '.')}

IMPORTANT RULES:
1. ALWAYS READ FILES instead of guessing or speculating:
   - If asked about a module/package → READ the actual files (__init__.py, main files)
   - If asked "how to use" → READ README.md and example files
   - If asked about functionality → SEARCH for the actual code
   - NEVER say "based on the structure" without reading files

2. Use MULTIPLE tools when appropriate:
   - "How to use X" → read_file(README), find_files(examples), read_file(main module)
   - "What does X do" → search_in_files(class/function), read_file(implementation)
   - "Show me the module" → list_directory(module path), read_file(main files)

3. Be PROACTIVE and THOROUGH:
   - Read actual implementation files, not just structure
   - Use multiple tools in parallel for comprehensive answers
   - Default to gathering MORE information rather than less

4. Parameter Requirements:
   - list_directory: directory_path (use "." for current)
   - read_file: file_path (exact path)
   - search_in_files: pattern (search text), file_pattern (optional)
   - find_files: pattern (filename pattern)
   - fuzzy_find: query (search term)

Decide which tools to use. Be helpful and proactive in gathering information."""

        try:
            # Use structured LLM response with Pydantic models for exact parameter validation
            from pydantic import BaseModel, Field
            from typing import List, Optional, Literal, Union
            
            # Define exact parameter models for each tool
            class ReadFileParams(BaseModel):
                file_path: str = Field(description="Path to the file to read")
                start_line: Optional[int] = Field(None, description="Starting line number")
                end_line: Optional[int] = Field(None, description="Ending line number")
                encoding: Optional[str] = Field("utf-8", description="File encoding")
            
            class ListDirectoryParams(BaseModel):
                directory_path: str = Field(".", description="Path to directory (use '.' for current)")
                pattern: Optional[str] = Field(None, description="File pattern filter")
                recursive: Optional[bool] = Field(False, description="List recursively")
                include_hidden: Optional[bool] = Field(False, description="Include hidden files")
            
            class SearchInFilesParams(BaseModel):
                pattern: str = Field(description="Text pattern to search for")
                file_pattern: Optional[str] = Field(None, description="File pattern to search in")
                max_results: Optional[int] = Field(50, description="Maximum results")
                case_sensitive: Optional[bool] = Field(False, description="Case sensitive search")
            
            class FindFilesParams(BaseModel):
                pattern: str = Field(description="File name pattern to search for")
                recursive: Optional[bool] = Field(True, description="Search recursively")
                include_hidden: Optional[bool] = Field(False, description="Include hidden files")
            
            class FuzzyFindParams(BaseModel):
                query: str = Field(description="Fuzzy search query")
                max_results: Optional[int] = Field(10, description="Maximum results")
            
            # Union type for all possible parameter models
            ToolParams = Union[ReadFileParams, ListDirectoryParams, SearchInFilesParams, FindFilesParams, FuzzyFindParams]
            
            class ToolSpec(BaseModel):
                name: Literal["read_file", "list_directory", "search_in_files", "find_files", "fuzzy_find"]
                params: ToolParams = Field(description="Tool-specific parameters with correct types")
                reason: str = Field(description="Why this tool is needed for the query")
            
            class ToolSelection(BaseModel):
                should_use_tools: bool = Field(description="Whether any tools should be executed")
                tools: List[ToolSpec] = Field(
                    default_factory=list,
                    description="List of tools to execute with validated parameters"
                )
                reasoning: str = Field(description="Explanation of tool selection decision")
            
            # Enhanced prompt - Pydantic models will enforce correct parameter structure
            enhanced_prompt = f"""{prompt}

EXAMPLES OF WHEN TO USE TOOLS:

Query: "tell me about the folder" or "what's in this directory"
→ USE: list_directory with directory_path="."

Query: "show me the README" or "what's in config.py"
→ USE: read_file with file_path="README.md" or fuzzy_find with query="README"

Query: "find all test files"
→ USE: find_files with pattern="*test*.py"

Query: "search for database connections"
→ USE: search_in_files with pattern="database" or pattern="connection"

Query: "hello" or "how are you"
→ NO TOOLS NEEDED (greeting)

RESPONSE FORMAT:
{{
  "should_use_tools": true,
  "tools": [
    {{
      "name": "list_directory",
      "params": {{"directory_path": "."}},
      "reason": "User asked about the folder/directory"
    }}
  ],
  "reasoning": "User wants to know about the folder structure"
}}

For the current query "{query}", select tools that will help gather relevant information."""
            
            # Get LLM response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                llm_client.generate,
                enhanced_prompt,
                ToolSelection
            )
            
            if response.should_use_tools and response.tools:
                # Convert Pydantic models to tool execution format
                tools_to_execute = []
                
                for tool_spec in response.tools:
                    # Convert Pydantic model to dict, filtering out None values for optional params
                    if hasattr(tool_spec.params, 'model_dump'):
                        # Pydantic v2
                        params_dict = tool_spec.params.model_dump(exclude_none=True)
                    else:
                        # Pydantic v1 
                        params_dict = tool_spec.params.dict(exclude_none=True)
                    
                    tools_to_execute.append({
                        'name': tool_spec.name,
                        'params': params_dict,
                        'timeout': 5.0
                    })
                
                logger.info(f"LLM selected {len(tools_to_execute)} tools: {response.reasoning}")
                return tools_to_execute
            else:
                logger.info(f"LLM decided no tools needed: {response.reasoning}")
                return []
                
        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}")
            # Fallback to intent-based tool selection
            return self._fallback_tool_selection(intent, context)
    
    async def _llm_validate_and_enhance_tools(
        self, 
        query: str, 
        suggested_tools: List[Dict[str, Any]], 
        intent, 
        context: Dict[str, Any],
        llm_client
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to validate keyword-matched tools and potentially enhance them.
        
        This handles typos, adds missing tools, and ensures comprehensive exploration.
        """
        try:
            # Build validation prompt
            validation_prompt = f"""You are a tool selection validator. Review and enhance the tool selection.

Original Query: "{query}"
Query Intent: {intent.type.value if intent else 'unknown'}

Suggested Tools from Keywords:
{self._format_tools_for_prompt(suggested_tools)}

Available Tools:
- list_directory: List files/dirs (params: directory_path)
- read_file: Read file contents (params: file_path)  
- search_in_files: Search text (params: pattern, file_pattern)
- find_files: Find by name (params: pattern)
- fuzzy_find: Fuzzy search (params: query)

VALIDATION TASKS:
1. Check if suggested tools are appropriate for the query
2. Fix any typos or incorrect paths (e.g., "copilt" → "copilot")
3. Add missing tools that would help answer the query
4. Remove redundant or unnecessary tools
5. Ensure file paths are correct (e.g., src/nighty_code/...)

IMPORTANT:
- If user asks about a module, READ its actual files
- If user asks "how to use", READ documentation AND examples
- If user has typos, fix them (e.g., "nighty_cdoe" → "nighty_code")
- Be comprehensive - better to read more than less

Return the validated and enhanced tool list."""

            # Define response model with proper tool structure
            from pydantic import BaseModel, Field
            from typing import List, Dict, Any, Optional
            
            class ToolSpec(BaseModel):
                name: str = Field(description="Tool name (e.g., 'read_file', 'list_directory')")
                params: Dict[str, Any] = Field(description="Tool parameters")
                timeout: Optional[float] = Field(default=5.0, description="Execution timeout")
            
            class ValidatedTools(BaseModel):
                tools: List[ToolSpec] = Field(description="Validated and enhanced tool list")
                changes_made: str = Field(description="Brief explanation of changes")
                typos_fixed: List[str] = Field(default_factory=list, description="List of typos that were corrected")
            
            # Get LLM validation
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                llm_client.generate,
                validation_prompt,
                ValidatedTools
            )
            
            if response.typos_fixed:
                logger.info(f"Fixed typos: {', '.join(response.typos_fixed)}")
            
            logger.info(f"LLM validation: {response.changes_made}")
            
            # Convert ToolSpec objects back to dicts
            validated_tools = []
            for tool in response.tools:
                if hasattr(tool, 'model_dump'):
                    # Pydantic v2
                    validated_tools.append(tool.model_dump())
                elif hasattr(tool, 'dict'):
                    # Pydantic v1
                    validated_tools.append(tool.dict())
                else:
                    # Already a dict
                    validated_tools.append(tool)
            
            return validated_tools
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}, using original tools")
            return suggested_tools
    
    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools list for prompt."""
        formatted = []
        for tool in tools:
            params_str = ', '.join(f"{k}={v}" for k, v in tool.get('params', {}).items())
            formatted.append(f"- {tool['name']}({params_str})")
        return '\n'.join(formatted)
    
    def _should_use_chaining(self, query: str, intent) -> bool:
        """
        Determine if the query requires chained tool execution.
        
        Complex queries that need step-by-step exploration should use chaining.
        """
        query_lower = query.lower()
        
        # Queries that require understanding AND then action
        chaining_indicators = [
            'and how',  # "what is X and how to use it"
            'then show',  # "find X then show me how"
            'and explain',  # "read X and explain"
            'what is.*how',  # "what is copilot and how do I use it"
            'find.*and.*',  # "find the module and read its docs"
            'understand.*use',  # "understand the module and use it"
        ]
        
        import re
        for pattern in chaining_indicators:
            if re.search(pattern, query_lower):
                return True
        
        # Multi-part questions
        if ' and ' in query_lower and ('how' in query_lower or 'what' in query_lower or 'show' in query_lower):
            return True
        
        # Questions about understanding AND using something
        if ('what is' in query_lower or 'what does' in query_lower) and \
           ('how to' in query_lower or 'how do' in query_lower or 'use it' in query_lower):
            return True
        
        return False
    
    async def _execute_chained_tools(
        self,
        query: str,
        intent,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute tools in a chained manner where each step can use results from previous steps.
        
        This allows for intelligent exploration where the LLM plans multiple steps
        and adapts based on intermediate results.
        """
        # Get LLM client
        llm_client = getattr(self, 'main_llm_client', None)
        if not llm_client:
            llm_client = getattr(self.intent_recognizer, 'llm_client', None)
        
        if not llm_client:
            # Fall back to parallel execution
            logger.warning("No LLM available for chained execution, falling back to parallel")
            tools = await self._llm_select_tools(query, intent, context)
            if tools:
                results = await self.tool_executor.execute_multiple(tools, parallel=True)
                return [{'tools_executed': [r.tool_name for r in results],
                        'results': [{'tool': r.tool_name, 'status': r.status.value, 'data': r.data} for r in results]}]
            return []
        
        # Create execution plan
        execution_plan = await self._create_execution_plan(query, context, llm_client)
        
        chain_results = []
        accumulated_context = []
        
        # Execute plan step by step
        for step_num, step in enumerate(execution_plan.get('steps', []), 1):
            logger.info(f"Executing chain step {step_num}: {step.get('description', 'Processing')}")
            
            # Prepare step context with results from previous steps
            step_context = {
                'original_query': query,
                'current_step': step_num,
                'total_steps': len(execution_plan.get('steps', [])),
                'previous_results': accumulated_context,
                **context
            }
            
            # Get tools for this step
            step_tools = await self._get_step_tools(
                step, 
                step_context, 
                accumulated_context,
                llm_client
            )
            
            if step_tools:
                # Execute tools for this step
                step_results = await asyncio.wait_for(
                    self.tool_executor.execute_multiple(step_tools, parallel=True),
                    timeout=10.0
                )
                
                # Store results for next steps
                step_data = {
                    'step': step_num,
                    'description': step.get('description', ''),
                    'tools_executed': [r.tool_name for r in step_results],
                    'results': []
                }
                
                for result in step_results:
                    result_data = {
                        'tool': result.tool_name,
                        'status': result.status.value,
                        'data': result.data
                    }
                    step_data['results'].append(result_data)
                    
                    # Add to accumulated context for next steps
                    if result.status.value == 'success':
                        accumulated_context.append({
                            'step': step_num,
                            'tool': result.tool_name,
                            'data': result.data
                        })
                
                chain_results.append(step_data)
            
            # Check if we should continue or stop based on results
            if step.get('conditional', False):
                should_continue = await self._evaluate_continuation(
                    step, accumulated_context, query, llm_client
                )
                if not should_continue:
                    logger.info(f"Stopping chain execution at step {step_num} based on results")
                    break
        
        return chain_results
    
    async def _create_execution_plan(
        self,
        query: str,
        context: Dict[str, Any],
        llm_client
    ) -> Dict[str, Any]:
        """
        Use LLM to create a multi-step execution plan for the query.
        """
        planning_prompt = f"""Create a step-by-step execution plan to answer this query comprehensively.

Query: "{query}"
Context: {context.get('project_path', 'current project')}

Available Tools:
- list_directory: Explore directory structure
- read_file: Read specific file contents
- search_in_files: Search for patterns in files
- find_files: Find files by name pattern
- fuzzy_find: Fuzzy search for files

PLANNING GUIDELINES:
1. Break down complex queries into logical steps
2. Each step should build on previous results
3. Start with exploration/discovery, then drill down to specifics
4. For "what is X and how to use it" queries:
   - Step 1: Find/explore the module
   - Step 2: Read main implementation files
   - Step 3: Find examples or documentation
   - Step 4: Synthesize usage patterns

Create a plan with 2-4 steps maximum. Each step should have:
- description: What this step accomplishes
- goal: What information we're seeking
- conditional: Whether to continue based on results (true/false)

Example for "what is copilot module and how do I use it":
Step 1: Explore copilot module structure
Step 2: Read main implementation files
Step 3: Find usage examples and documentation"""

        try:
            from pydantic import BaseModel, Field
            from typing import List, Optional
            
            class ExecutionStep(BaseModel):
                description: str = Field(description="What this step does")
                goal: str = Field(description="Information we're seeking")
                conditional: bool = Field(default=False, description="Whether continuation depends on results")
                suggested_tools: List[str] = Field(default_factory=list, description="Tools likely needed")
            
            class ExecutionPlan(BaseModel):
                steps: List[ExecutionStep] = Field(description="Ordered execution steps")
                strategy: str = Field(description="Overall strategy explanation")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                llm_client.generate,
                planning_prompt,
                ExecutionPlan
            )
            
            logger.info(f"Execution plan created: {response.strategy}")
            
            # Convert to dict
            if hasattr(response, 'model_dump'):
                plan = response.model_dump()
            elif hasattr(response, 'dict'):
                plan = response.dict()
            else:
                plan = {'steps': [], 'strategy': 'Default exploration'}
            
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to create execution plan: {e}")
            # Default plan
            return {
                'steps': [
                    {'description': 'Explore and understand', 'goal': 'Find relevant files', 'conditional': False},
                    {'description': 'Read and analyze', 'goal': 'Understand implementation', 'conditional': False}
                ],
                'strategy': 'Default exploration strategy'
            }
    
    async def _get_step_tools(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
        llm_client
    ) -> List[Dict[str, Any]]:
        """
        Determine which tools to use for a specific step based on previous results.
        """
        # Build context from previous results
        results_summary = self._summarize_results(previous_results)
        
        # PROACTIVE: If we just listed a directory, automatically read key files
        auto_tools = self._get_auto_followup_tools(previous_results)
        if auto_tools:
            logger.info(f"Auto-reading {len(auto_tools)} files discovered in previous step")
            return auto_tools
        
        prompt = f"""Based on the execution step and previous results, select tools to execute.

Current Step: {step.get('description', 'Processing')}
Step Goal: {step.get('goal', 'Gather information')}

Previous Results Summary:
{results_summary}

Available Tools:
- list_directory(directory_path): List files and directories
- read_file(file_path): Read file contents
- search_in_files(pattern, file_pattern): Search in files
- find_files(pattern): Find files by pattern
- fuzzy_find(query): Fuzzy file search

IMPORTANT: Be PROACTIVE!
- If you found files in a directory listing, READ THEM
- If you found Python files, read __init__.py, main files, and key implementations
- Don't just list files - actually read their contents
- If searching found matches, read those specific files

Return specific tools with exact parameters based on the discovered information."""

        try:
            from pydantic import BaseModel, Field
            from typing import List, Dict, Any
            
            class StepTools(BaseModel):
                tools: List[Dict[str, Any]] = Field(description="Tools to execute for this step")
                reasoning: str = Field(description="Why these tools were selected")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                llm_client.generate,
                prompt,
                StepTools
            )
            
            logger.info(f"Step tools selected: {response.reasoning}")
            
            # Ensure proper format
            tools = []
            for tool in response.tools:
                if isinstance(tool, dict) and 'name' in tool:
                    tools.append({
                        'name': tool['name'],
                        'params': tool.get('params', {}),
                        'timeout': tool.get('timeout', 5.0)
                    })
            
            return tools
            
        except Exception as e:
            logger.warning(f"Failed to select step tools: {e}")
            # Fallback to suggested tools from plan
            suggested = step.get('suggested_tools', [])
            if suggested:
                return [{'name': t, 'params': {}, 'timeout': 5.0} for t in suggested]
            return []
    
    def _get_auto_followup_tools(self, previous_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Automatically generate follow-up tools based on previous results.
        If we just listed a directory, read the key files automatically.
        """
        if not previous_results:
            return []
        
        auto_tools = []
        
        # Check the most recent result
        for result in previous_results[-1:]:  # Last result
            if result.get('tool') == 'list_directory' and result.get('data'):
                data = result['data']
                if isinstance(data, dict):
                    files = data.get('files', [])
                    path = data.get('path', '.')
                    
                    # Priority files to read automatically
                    priority_files = ['__init__.py', 'index.py', 'main.py', 'server.py', 'client.py', 'base.py']
                    readme_files = ['README.md', 'README.txt', 'readme.md']
                    
                    # Read priority Python files
                    for file in files:
                        if file in priority_files:
                            file_path = f"{path}/{file}" if path != '.' else file
                            auto_tools.append({
                                'name': 'read_file',
                                'params': {'file_path': file_path},
                                'timeout': 5.0
                            })
                    
                    # Read README if present
                    for file in files:
                        if file in readme_files and len(auto_tools) < 5:  # Limit to 5 files
                            file_path = f"{path}/{file}" if path != '.' else file
                            auto_tools.append({
                                'name': 'read_file',
                                'params': {'file_path': file_path},
                                'timeout': 5.0
                            })
                    
                    # If no priority files, read first 3 Python files
                    if not auto_tools:
                        py_files = [f for f in files if f.endswith('.py')]
                        for file in py_files[:3]:
                            file_path = f"{path}/{file}" if path != '.' else file
                            auto_tools.append({
                                'name': 'read_file',
                                'params': {'file_path': file_path},
                                'timeout': 5.0
                            })
        
        return auto_tools
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize previous results for context."""
        if not results:
            return "No previous results"
        
        summary = []
        for item in results[-3:]:  # Last 3 results for context
            tool = item.get('tool', 'unknown')
            data = item.get('data', {})
            
            if tool == 'list_directory' and isinstance(data, dict):
                files = data.get('files', [])
                dirs = data.get('directories', [])
                summary.append(f"Found {len(files)} files and {len(dirs)} directories: {', '.join(files[:5])}")
            
            elif tool == 'read_file':
                if isinstance(data, str):
                    summary.append(f"Read file content ({len(data)} chars)")
                elif isinstance(data, dict):
                    summary.append(f"Read file: {data.get('path', 'unknown')}")
            
            elif tool == 'find_files' and isinstance(data, dict):
                matches = data.get('matches', [])
                summary.append(f"Found {len(matches)} matching files")
            
            elif tool == 'search_in_files' and isinstance(data, dict):
                matches = data.get('matches', [])
                summary.append(f"Found {len(matches)} search matches")
        
        return '\n'.join(summary) if summary else "Previous steps completed"
    
    async def _evaluate_continuation(
        self,
        step: Dict[str, Any],
        results: List[Dict[str, Any]],
        query: str,
        llm_client
    ) -> bool:
        """
        Evaluate whether to continue execution based on current results.
        """
        # For now, always continue unless we found nothing
        if not results:
            return False
        
        # Check if we have enough information
        has_files = any('read_file' in r.get('tool', '') for r in results)
        has_listing = any('list_directory' in r.get('tool', '') for r in results)
        
        return has_files or has_listing
    
    def _fallback_tool_selection(self, intent, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback tool selection when LLM is not available."""
        from .intent import IntentType
        
        tools = []
        
        # Simple mapping based on intent type
        if intent.type == IntentType.FILE_READ:
            # Look for file entities
            for entity in intent.entities:
                if entity.type.value in ['file_path', 'file_name']:
                    tools.append({
                        'name': 'read_file',
                        'params': {'file_path': entity.value},
                        'timeout': 5.0
                    })
                    break
            else:
                # No specific file, try fuzzy find with first entity
                if intent.entities:
                    tools.append({
                        'name': 'fuzzy_find', 
                        'params': {'query': intent.entities[0].value},
                        'timeout': 5.0
                    })
        
        elif intent.type == IntentType.DIR_LIST:
            # Look for directory entity or use current
            dir_path = '.'
            for entity in intent.entities:
                if entity.type.value == 'dir_path':
                    dir_path = entity.value
                    break
            
            tools.append({
                'name': 'list_directory',
                'params': {'directory_path': dir_path},
                'timeout': 5.0
            })
        
        elif intent.type == IntentType.FILE_SEARCH:
            # Use first entity as search query
            if intent.entities:
                tools.append({
                    'name': 'search_files',
                    'params': {'query': intent.entities[0].value},
                    'timeout': 5.0
                })
        
        elif intent.type == IntentType.CODE_SEARCH:
            # Look for pattern entity
            for entity in intent.entities:
                if entity.type.value == 'code_pattern':
                    tools.append({
                        'name': 'search_pattern',
                        'params': {'pattern': entity.value},
                        'timeout': 5.0
                    })
                    break
        
        return tools
    
    def validate_path(self, path: str) -> bool:
        """
        Validate a file or directory path.
        
        Args:
            path: Path to validate
            
        Returns:
            Whether the path is valid
        """
        if not self.validator:
            return True
        
        try:
            self.validator.validate_path(path)
            return True
        except ValueError:
            return False