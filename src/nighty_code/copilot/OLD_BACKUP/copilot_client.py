"""
Production-ready Copilot Client with comprehensive error handling,
monitoring, and graceful degradation.
"""

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from dotenv import load_dotenv

from ..llm.client import LLMClient
from ..llm.config import LLMConfig, LLMProvider, TokenStrategy
from ..llm.structured_client import StructuredLLMClient
from .persona import CopilotPersona, get_persona
from .memory import CopilotMemory, MemoryManager
from .session import SessionManager, SessionStatus
from .robust_intent import (
    RobustIntentRecognizer,
    RecognizerConfig,
    MetricsCollector,
    AuditLogger
)
from .robust_tool_chain import RobustToolChain, ToolChainConfig
from ..mcp.server import MCPServer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CopilotConfig:
    """Configuration for production copilot."""
    
    def __init__(self):
        # Load from environment or use defaults
        self.enable_mcp = os.getenv("COPILOT_ENABLE_MCP", "true").lower() == "true"
        self.enable_metrics = os.getenv("COPILOT_ENABLE_METRICS", "true").lower() == "true"
        self.enable_audit = os.getenv("COPILOT_ENABLE_AUDIT", "true").lower() == "true"
        self.enable_caching = os.getenv("COPILOT_ENABLE_CACHING", "true").lower() == "true"
        
        # Timeouts
        self.default_timeout = float(os.getenv("COPILOT_DEFAULT_TIMEOUT", "30.0"))
        self.tool_timeout = float(os.getenv("COPILOT_TOOL_TIMEOUT", "15.0"))
        self.llm_timeout = float(os.getenv("COPILOT_LLM_TIMEOUT", "20.0"))
        
        # Limits
        self.max_tool_retries = int(os.getenv("COPILOT_MAX_TOOL_RETRIES", "3"))
        self.max_concurrent_tools = int(os.getenv("COPILOT_MAX_CONCURRENT_TOOLS", "5"))
        
        # Feature flags
        self.graceful_degradation = os.getenv("COPILOT_GRACEFUL_DEGRADATION", "true").lower() == "true"
        self.auto_recovery = os.getenv("COPILOT_AUTO_RECOVERY", "true").lower() == "true"


class CopilotClient:
    """
    Production-ready copilot client with comprehensive error handling,
    monitoring, and graceful degradation.
    """
    
    def __init__(
        self,
        folder_path: Optional[str] = None,
        persona_type: str = "default",
        llm_config: Optional[LLMConfig] = None,
        config: Optional[CopilotConfig] = None
    ):
        """
        Initialize production copilot client.
        
        Args:
            folder_path: Path to project folder
            persona_type: Type of persona to use
            llm_config: LLM configuration
            config: Production configuration
        """
        self.config = config or CopilotConfig()
        self.request_counter = 0
        
        # Initialize project path with validation
        try:
            if folder_path:
                self.project_path = Path(folder_path).resolve()
                if not self.project_path.exists():
                    logger.warning(f"Project path does not exist: {folder_path}, using current directory")
                    self.project_path = Path.cwd()
                elif not self.project_path.is_dir():
                    raise ValueError(f"Project path is not a directory: {folder_path}")
            else:
                self.project_path = Path.cwd()
        except Exception as e:
            logger.error(f"Failed to initialize project path: {e}")
            self.project_path = Path.cwd()
        
        logger.info(f"Initializing CopilotClient for project: {self.project_path}")
        
        # Initialize persona
        try:
            self.persona = get_persona(persona_type)
        except Exception as e:
            logger.error(f"Failed to load persona {persona_type}: {e}")
            self.persona = get_persona("default")  # Fallback to default
        
        # Initialize project understanding
        self.project_context = None
        try:
            self._analyze_project()
        except Exception as e:
            logger.error(f"Failed to analyze project: {e}")
            self.project_context = f"Project: {self.project_path.name}"
        
        # Initialize LLM client with error handling
        try:
            if llm_config is None:
                llm_config = self._create_default_config()
            self.llm_client = LLMClient(llm_config)
            self.structured_llm_client = StructuredLLMClient(llm_config)
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
            self.structured_llm_client = None
            if not self.config.graceful_degradation:
                raise
        
        # Initialize memory and session management
        try:
            self.memory_manager = MemoryManager(self.llm_client) if self.llm_client else None
            self.session_manager = SessionManager()
            self.current_memory: Optional[CopilotMemory] = None
            self.current_session = None
        except Exception as e:
            logger.error(f"Failed to initialize memory/session management: {e}")
            self.memory_manager = None
            self.session_manager = None
        
        # Generate system prompt
        self._generate_system_prompt()
        
        # Initialize MCP integration if enabled
        if self.config.enable_mcp:
            try:
                self._initialize_mcp()
            except Exception as e:
                logger.error(f"Failed to initialize MCP: {e}")
                if not self.config.graceful_degradation:
                    raise
                self.mcp_server = None
                self.intent_recognizer = None
                self.tool_chain = None
        else:
            self.mcp_server = None
            self.intent_recognizer = None
            self.tool_chain = None
        
        # Initialize metrics and audit
        if self.config.enable_metrics:
            self.metrics = MetricsCollector()
        else:
            self.metrics = None
        
        if self.config.enable_audit:
            self.audit = AuditLogger()
        else:
            self.audit = None
        
        logger.info(f"CopilotClient initialized successfully")
    
    def _initialize_mcp(self):
        """Initialize MCP components with error handling."""
        try:
            # Initialize MCP server
            self.mcp_server = MCPServer(self.project_path)
            
            # Configure intent recognizer
            recognizer_config = RecognizerConfig(
                enable_llm_fallback=self.structured_llm_client is not None,
                enable_caching=self.config.enable_caching,
                enable_metrics=self.config.enable_metrics,
                enable_audit_logging=self.config.enable_audit,
                llm_timeout=self.config.llm_timeout
            )
            
            # Initialize intent recognizer
            self.intent_recognizer = RobustIntentRecognizer(
                config=recognizer_config,
                llm_client=self.structured_llm_client,
                project_root=self.project_path
            )
            
            # Configure tool chain
            tool_config = ToolChainConfig(
                max_retries=self.config.max_tool_retries,
                default_timeout=self.config.tool_timeout,
                max_concurrent=self.config.max_concurrent_tools,
                enable_recovery=self.config.auto_recovery
            )
            
            # Initialize tool chain
            self.tool_chain = RobustToolChain(
                mcp_server=self.mcp_server,
                config=tool_config,
                metrics=self.metrics,
                audit=self.audit
            )
            
            logger.info("MCP components initialized successfully")
            
        except Exception as e:
            logger.error(f"MCP initialization error: {e}")
            raise
    
    def _create_default_config(self) -> LLMConfig:
        """Create default LLM configuration with validation."""
        try:
            # Determine provider
            provider_str = os.getenv("LLM_PROVIDER", "anthropic").lower()
            if provider_str == "anthropic":
                provider = LLMProvider.ANTHROPIC
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider_str == "openai":
                provider = LLMProvider.OPENAI
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError(f"Unsupported LLM provider: {provider_str}")
            
            if not api_key:
                raise ValueError(f"API key not found for provider: {provider_str}")
            
            # Get model and settings
            model = os.getenv("LLM_MODEL", 
                            "claude-3-5-haiku-20241022" if provider == LLMProvider.ANTHROPIC else "gpt-3.5-turbo")
            
            return LLMConfig(
                provider=provider,
                api_key=api_key,
                model=model,
                input_token_strategy=TokenStrategy.TRUNCATE,
                output_token_strategy=TokenStrategy.CONTINUE,
                track_costs=True,
                cost_warning_threshold=float(os.getenv("LLM_COST_WARNING_THRESHOLD", "1.0")),
                log_requests=os.getenv("LLM_LOG_REQUESTS", "false").lower() == "true",
                log_responses=os.getenv("LLM_LOG_RESPONSES", "false").lower() == "true",
                log_tokens=os.getenv("LLM_LOG_TOKENS", "true").lower() == "true"
            )
        except Exception as e:
            logger.error(f"Failed to create LLM config: {e}")
            raise
    
    def _analyze_project(self):
        """Analyze project structure with error handling."""
        try:
            logger.info("Analyzing project structure...")
            
            # Simple progressive exploration
            exploration_result = self._progressive_explore(self.project_path, token_budget=2000)
            
            if exploration_result['content']:
                self.project_context = "\n\n".join(exploration_result['content'])
            else:
                self.project_context = f"Project: {self.project_path.name}\nLocation: {self.project_path}"
            
            logger.info(f"Project analysis complete: {exploration_result['file_count']} files, "
                       f"{exploration_result['dir_count']} directories")
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            self.project_context = f"Project: {self.project_path.name}"
    
    def _progressive_explore(self, path: Path, token_budget: int = 4000) -> Dict:
        """Progressive folder exploration with error handling."""
        result = {
            'content': [],
            'file_count': 0,
            'dir_count': 0,
            'tokens_used': 0
        }
        
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        def add_content_if_budget_allows(content: str) -> bool:
            tokens_needed = estimate_tokens(content)
            if result['tokens_used'] + tokens_needed <= token_budget:
                result['content'].append(content)
                result['tokens_used'] += tokens_needed
                return True
            return False
        
        try:
            # Collect items safely
            items = []
            try:
                for item in path.iterdir():
                    if item.name.startswith('.'):
                        continue
                    items.append(item)
                    if item.is_dir():
                        result['dir_count'] += 1
                    else:
                        result['file_count'] += 1
            except PermissionError:
                logger.warning(f"Permission denied accessing {path}")
                return result
            
            # Sort by priority
            def sort_key(item):
                if item.name.lower().startswith('readme'):
                    return (0, item.name)
                elif item.is_dir() and item.name in ['src', 'lib', 'app', 'core']:
                    return (1, item.name)
                elif item.is_dir():
                    return (2, item.name)
                else:
                    return (3, item.name)
            
            items.sort(key=sort_key)
            
            # Basic structure
            dirs = [item.name for item in items if item.is_dir()][:20]  # Limit
            files = [item.name for item in items if item.is_file()][:20]  # Limit
            
            structure_desc = f"Project: {path.name}\nLocation: {path}\n\n"
            if dirs:
                structure_desc += f"Directories ({len(dirs)}): {', '.join(dirs)}\n"
            if files:
                structure_desc += f"Files ({len(files)}): {', '.join(files)}\n"
            
            add_content_if_budget_allows(structure_desc)
            
            # Add priority files
            priority_files = ['README.md', 'README.txt', 'package.json', 'requirements.txt', 'setup.py']
            
            for item in items:
                if result['tokens_used'] >= token_budget:
                    break
                
                if item.is_file() and item.name in priority_files:
                    try:
                        content = item.read_text(encoding='utf-8', errors='ignore')[:1000]
                        file_content = f"\n--- {item.name} ---\n{content}\n"
                        add_content_if_budget_allows(file_content)
                    except Exception as e:
                        logger.debug(f"Could not read {item.name}: {e}")
                        continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error in progressive exploration: {e}")
            result['content'] = [f"Project: {path.name}\nExploration failed: {str(e)}"]
            return result
    
    def _generate_system_prompt(self):
        """Generate system prompt with project context."""
        try:
            base_prompt = self.persona.generate_system_prompt(str(self.project_path))
            
            if self.project_context:
                self.system_prompt = f"""{base_prompt}

## PROJECT CONTEXT

{self.project_context}

---

You have knowledge about this project's structure. Use this context to provide accurate assistance."""
            else:
                self.system_prompt = base_prompt
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            self.system_prompt = "You are a helpful coding assistant."
    
    async def ask(
        self,
        question: str,
        use_memory: bool = True,
        use_tools: bool = True,
        timeout: Optional[float] = None
    ) -> str:
        """
        Ask the copilot a question with production safeguards.
        
        Args:
            question: The question to ask
            use_memory: Whether to use conversation memory
            use_tools: Whether to use MCP tools
            timeout: Optional timeout override
            
        Returns:
            The copilot's response
        """
        request_id = str(uuid.uuid4())
        timeout = timeout or self.config.default_timeout
        
        # Log request
        if self.audit:
            self.audit.log_request(request_id, question, None)
        
        try:
            # Apply timeout to entire operation
            async with asyncio.timeout(timeout):
                # Use MCP tools if enabled
                tool_context = None
                if use_tools and self.intent_recognizer and self.tool_chain:
                    try:
                        tool_context = await self._execute_with_tools(question, request_id)
                    except Exception as e:
                        logger.warning(f"Tool execution failed: {e}")
                        # Continue without tools if graceful degradation enabled
                        if not self.config.graceful_degradation:
                            raise
                
                # Enhance question with tool context
                if tool_context:
                    question = self._enhance_with_tool_context(question, tool_context)
                
                # Prepare context with memory
                full_context = self.system_prompt
                if use_memory and self.current_memory:
                    memory_context = self.current_memory.get_formatted_context()
                    if memory_context:
                        full_context = f"{self.system_prompt}\n\n{memory_context}"
                
                # Get LLM response
                if self.llm_client:
                    response = await asyncio.to_thread(
                        self.llm_client.complete,
                        prompt=question,
                        system_prompt=full_context,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    answer = response.content
                    
                    # Update memory
                    if use_memory and self.current_memory:
                        self.current_memory.add_exchange(question, answer)
                    
                    # Record metrics
                    if self.metrics:
                        self.metrics.record_request("ask", 0, True)
                    
                    return answer
                else:
                    # No LLM available
                    return "I apologize, but I'm unable to process your request at the moment. The LLM service is unavailable."
                    
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            if self.metrics:
                self.metrics.record_error("timeout")
            return "I apologize, but the request timed out. Please try again with a simpler question."
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            if self.metrics:
                self.metrics.record_error(type(e).__name__)
            return self.persona.format_error_response(e) if self.persona else f"An error occurred: {e}"
    
    async def _execute_with_tools(self, query: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute MCP tools based on query with error handling.
        
        Args:
            query: User's query
            request_id: Request ID for tracing
            
        Returns:
            Tool execution context or None
        """
        try:
            # Build context
            context = {
                'project_path': str(self.project_path),
                'current_directory': str(self.project_path)
            }
            
            # Add memory context if available
            if self.current_memory:
                try:
                    recent_exchanges = self.current_memory.get_recent_exchanges(2)
                    if recent_exchanges:
                        context['recent_files'] = []
                        for ex in recent_exchanges:
                            files = re.findall(r'[\w/]+\.\w+', ex.get('response', ''))
                            context['recent_files'].extend(files[:5])  # Limit
                except Exception as e:
                    logger.debug(f"Failed to extract memory context: {e}")
            
            # Recognize intent
            intent = await self.intent_recognizer.recognize(query, context, request_id)
            
            # Show understanding to user
            self._show_intent_understanding(intent)
            
            # Execute tools if confident enough
            if intent.confidence >= 0.3 and intent.suggested_tools:
                results = await self.tool_chain.execute_from_intent(intent, context, request_id)
                
                # Extract successful results
                tool_data = {}
                for result in results:
                    if result.success:
                        tool_data[result.tool_name] = result.data
                
                return tool_data if tool_data else None
            
            return None
            
        except Exception as e:
            logger.error(f"Tool execution failed for {request_id}: {e}")
            if not self.config.graceful_degradation:
                raise
            return None
    
    def _show_intent_understanding(self, intent):
        """Show intent understanding to user."""
        try:
            if intent.confidence >= 0.7:
                print(f"\n[Understanding: {intent.reasoning}]")
                if intent.ambiguous:
                    print("[Note: Query has some ambiguity]")
            elif intent.confidence >= 0.5:
                print(f"\n[Interpreting: {intent.reasoning}]")
            elif intent.confidence >= 0.3:
                print(f"\n[Exploring: {intent.reasoning}]")
            else:
                print("\n[Processing query...]")
        except Exception:
            pass  # Don't fail on display errors
    
    def _enhance_with_tool_context(self, question: str, tool_context: Dict[str, Any]) -> str:
        """Enhance question with tool execution results."""
        try:
            enhanced = question + "\n\n### CONTEXT FROM PROJECT EXPLORATION:\n"
            
            for tool_name, data in tool_context.items():
                if tool_name == 'read_file' and data:
                    enhanced += f"\nFile content from {data.get('path', 'unknown')}:\n"
                    content = str(data.get('content', ''))[:2000]
                    enhanced += content + "\n"
                
                elif tool_name == 'list_directory' and data:
                    enhanced += f"\nDirectory listing:\n"
                    files = data.get('files', [])[:10]
                    dirs = data.get('directories', [])[:10]
                    if dirs:
                        enhanced += f"Directories: {', '.join(dirs)}\n"
                    if files:
                        enhanced += f"Files: {', '.join(files)}\n"
                
                elif tool_name == 'search_files' and data:
                    results = data.get('results', [])[:5]
                    if results:
                        enhanced += f"\nSearch results: {len(results)} matches\n"
                        for result in results:
                            enhanced += f"  - {result}\n"
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to enhance context: {e}")
            return question
    
    def chat(self, resume_session: Optional[str] = None):
        """Start interactive chat session with error handling."""
        try:
            # Create or resume session
            if resume_session:
                session = self.session_manager.recover_session(resume_session) if self.session_manager else None
                if not session:
                    print(f"Could not resume session: {resume_session}")
                    return
                print(f"\nResumed session: {resume_session}")
            else:
                if self.session_manager:
                    session = self.session_manager.create_session(
                        project_path=str(self.project_path),
                        persona_type=self.persona.name
                    )
                    print(f"\nStarted new session: {session.session_id}")
                else:
                    session = None
                    print("\nStarted chat session (no persistence)")
            
            self.current_session = session
            
            # Create memory for session
            if self.memory_manager:
                self.current_memory = self.memory_manager.create_memory(
                    session_id=session.session_id if session else "temp",
                    max_tokens=6000
                )
            
            print("\n" + "="*60)
            print(f"  Copilot - {self.project_path.name}")
            if session:
                print(f"  Session: {session.session_id}")
            print(f"  Persona: {self.persona.name}")
            print("="*60)
            print(f"\n{self.persona.get_greeting()}")
            print("\nType 'exit' to quit. Commands: /stats, /health\n")
            
            # Main chat loop
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("\nGoodbye!")
                        break
                    
                    if user_input.startswith('/'):
                        self._handle_command(user_input)
                        continue
                    
                    # Get response
                    print("\nCopilot: ", end="", flush=True)
                    response = asyncio.run(self.ask(user_input, use_memory=True, use_tools=True))
                    
                    for line in response.split('\n'):
                        print(line)
                        
                except KeyboardInterrupt:
                    print("\n\nSession ended.")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    if not self.config.graceful_degradation:
                        break
                        
        except Exception as e:
            logger.error(f"Chat session error: {e}")
            print(f"\nFatal error: {e}")
        
        finally:
            # Save session
            if self.current_memory and self.current_session and self.session_manager:
                try:
                    self.session_manager.end_session(SessionStatus.COMPLETED)
                    print(f"\nSession saved: {self.current_session.session_id}")
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")
            
            self.current_memory = None
            self.current_session = None
    
    def _handle_command(self, command: str):
        """Handle special commands."""
        cmd = command.lower().strip()
        
        if cmd == '/stats':
            if self.metrics:
                metrics = self.metrics.get_metrics()
                print("\n=== Metrics ===")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("Metrics not enabled")
        
        elif cmd == '/health':
            print("\n=== Health Check ===")
            print(f"LLM Client: {'OK' if self.llm_client else 'UNAVAILABLE'}")
            print(f"MCP Server: {'OK' if self.mcp_server else 'UNAVAILABLE'}")
            if self.intent_recognizer:
                health = self.intent_recognizer.health_check()
                print(f"Intent Recognizer: {health['status']}")
                print(f"Circuit Breaker: {health['circuit_breaker_state']}")
        
        else:
            print(f"Unknown command: {command}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = {}
        
        if self.llm_client:
            stats['llm'] = self.llm_client.get_stats()
        
        if self.metrics:
            stats['metrics'] = self.metrics.get_metrics()
        
        if self.intent_recognizer:
            stats['intent_recognizer'] = self.intent_recognizer.get_metrics()
        
        return stats


# Export main class
__all__ = ["CopilotClient"]