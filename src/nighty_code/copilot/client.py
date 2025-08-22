"""
Main CopilotClient - unified implementation with clean architecture.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from ..llm.client import LLMClient
from ..llm.config import LLMConfig, LLMProvider, TokenStrategy
from ..llm.structured_client import StructuredLLMClient
from ..mcp.server import MCPServer

from .core.orchestrator import QueryOrchestrator
from .memory.manager import CopilotMemory, MemoryManager
from .memory.session import SessionManager, SessionStatus
from .models.persona import CopilotPersona, get_persona

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CopilotClient:
    """
    Main copilot client for AI-assisted code analysis.
    
    Provides an intelligent interface for understanding and navigating codebases.
    """
    
    def __init__(
        self,
        folder_path: Optional[str] = None,
        persona_type: str = "default",
        llm_config: Optional[LLMConfig] = None,
        use_mcp: bool = True,
        enable_validation: bool = True
    ):
        """
        Initialize the copilot client.
        
        Args:
            folder_path: Path to the project folder. Uses current directory if None.
            persona_type: Type of persona (default, architect, security, performance)
            llm_config: Optional LLM configuration. Uses environment defaults if None.
            use_mcp: Whether to enable MCP tool integration
            enable_validation: Whether to validate user inputs
        """
        # Set project path
        if folder_path:
            self.project_path = Path(folder_path).resolve()
            if not self.project_path.exists():
                raise ValueError(f"Project path does not exist: {folder_path}")
            if not self.project_path.is_dir():
                raise ValueError(f"Project path is not a directory: {folder_path}")
        else:
            self.project_path = Path.cwd()
        
        logger.info(f"Initializing CopilotClient for project: {self.project_path}")
        
        # Initialize persona
        self.persona = get_persona(persona_type)
        
        # Analyze project structure
        self.project_context = self._analyze_project()
        
        # Initialize LLM client
        if llm_config is None:
            llm_config = self._create_default_config()
        
        self.llm_client = LLMClient(llm_config)
        
        # Initialize structured LLM client for intent recognition
        self.structured_client = StructuredLLMClient(self.llm_client) if self.llm_client else None
        
        # Generate system prompt with project context
        self.system_prompt = self._generate_system_prompt()
        
        # Initialize MCP and orchestrator
        self.use_mcp = use_mcp
        if self.use_mcp:
            self.mcp_server = MCPServer(self.project_path)
        else:
            self.mcp_server = None
        
        # Initialize query orchestrator
        self.orchestrator = QueryOrchestrator(
            mcp_server=self.mcp_server,
            llm_client=self.structured_client,
            project_path=self.project_path,
            enable_validation=enable_validation,
            enable_tools=use_mcp
        )
        
        # Give orchestrator access to main LLM client for tool selection
        self.orchestrator.main_llm_client = self.structured_client
        
        # Memory and session management
        self.memory_manager = MemoryManager(self.llm_client)
        self.session_manager = SessionManager()
        self.current_memory: Optional[CopilotMemory] = None
        self.current_session = None
        
        logger.info(f"CopilotClient initialized with persona: {self.persona.name}, MCP: {self.use_mcp}")
    
    def _create_default_config(self) -> LLMConfig:
        """Create default LLM configuration from environment variables."""
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
            raise ValueError(f"API key not found for provider: {provider_str}. Please set {provider_str.upper()}_API_KEY in your environment.")
        
        # Get model and settings
        model = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022" if provider == LLMProvider.ANTHROPIC else "gpt-3.5-turbo")
        
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            input_token_strategy=TokenStrategy.TRUNCATE,
            output_token_strategy=TokenStrategy.CONTINUE,
            track_costs=os.getenv("LLM_TRACK_COSTS", "true").lower() == "true",
            log_tokens=os.getenv("LLM_LOG_TOKENS", "true").lower() == "true"
        )
    
    def _analyze_project(self) -> str:
        """Analyze project structure for context."""
        try:
            print("Analyzing project structure...")
            result = self._progressive_explore(self.project_path, token_budget=2000)
            print(f"  Found {result['file_count']} files, {result['dir_count']} directories")
            print(f"  Used {result['tokens_used']} tokens")
            
            if result['content']:
                context = "\n\n".join(result['content'])
                print("  Project context loaded")
                print("  Project analysis complete!")
                return context
            else:
                print("  Using basic fallback context")
                return f"Project: {self.project_path.name}\nLocation: {self.project_path}"
                
        except Exception as e:
            logger.error(f"Failed to analyze project: {e}")
            return f"Project: {self.project_path.name}\nLocation: {self.project_path}"
    
    def _progressive_explore(self, path: Path, token_budget: int = 2000) -> Dict:
        """Progressive folder exploration within token budget."""
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
            # Collect and prioritize items
            items = []
            for item in path.iterdir():
                if item.name.startswith('.'):
                    continue
                items.append(item)
                if item.is_dir():
                    result['dir_count'] += 1
                else:
                    result['file_count'] += 1
            
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
            
            # Basic structure overview
            dirs = [item.name for item in items if item.is_dir()]
            files = [item.name for item in items if item.is_file()]
            
            structure_desc = f"Project: {path.name}\nLocation: {path}\n\n"
            if dirs:
                structure_desc += f"Directories ({len(dirs)}): {', '.join(dirs)}\n"
            if files:
                structure_desc += f"Files ({len(files)}): {', '.join(files)}\n"
            
            if not add_content_if_budget_allows(structure_desc):
                return result
            
            # Add priority content within budget
            priority_files = ['README.md', 'README.txt', 'README', 'package.json', 'requirements.txt']
            
            for item in items:
                if result['tokens_used'] >= token_budget:
                    break
                
                # Priority files
                if item.is_file() and (item.name in priority_files or item.name.lower().startswith('readme')):
                    try:
                        content = item.read_text(encoding='utf-8', errors='ignore')[:1500]
                        file_content = f"\n--- {item.name} ---\n{content}\n"
                        add_content_if_budget_allows(file_content)
                    except Exception:
                        continue
                
                # Key directory listings
                elif item.is_dir() and item.name in ['src', 'lib', 'app', 'core']:
                    try:
                        subfiles = [f.name for f in item.iterdir() if f.is_file() and not f.name.startswith('.')][:8]
                        subdirs = [d.name for d in item.iterdir() if d.is_dir() and not d.name.startswith('.')][:8]
                        
                        dir_desc = f"\n--- {item.name}/ ---\n"
                        if subdirs:
                            dir_desc += f"Subdirectories: {', '.join(subdirs)}\n"
                        if subfiles:
                            dir_desc += f"Files: {', '.join(subfiles)}\n"
                        
                        add_content_if_budget_allows(dir_desc)
                    except Exception:
                        continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error in project exploration: {e}")
            result['content'] = [f"Project: {path.name}\nLocation: {path}\nExploration failed: {str(e)}"]
            result['tokens_used'] = estimate_tokens(result['content'][0])
            return result
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt with project context."""
        base_prompt = self.persona.generate_system_prompt(str(self.project_path))
        
        if self.project_context:
            return f"""{base_prompt}

## PROJECT CONTEXT

{self.project_context}

---

You have comprehensive knowledge about this project's structure, technologies, and purpose. 
Use this context to provide accurate, specific assistance."""
        else:
            return base_prompt
    
    def ask(self, question: str, use_memory: bool = True, use_tools: bool = True) -> str:
        """
        Ask the copilot a question about the codebase.
        
        Args:
            question: The question to ask
            use_memory: Whether to use conversation memory
            use_tools: Whether to use MCP tools for exploration
            
        Returns:
            The copilot's response
        """
        try:
            # Use orchestrator for tool execution if enabled
            if self.use_mcp and use_tools:
                try:
                    # Build context
                    context = {
                        'project_path': str(self.project_path),
                        'current_directory': str(self.project_path)
                    }
                    
                    # Add memory context if available
                    if use_memory and self.current_memory:
                        recent_exchanges = self.current_memory.get_recent_exchanges(2)
                        if recent_exchanges:
                            context['recent_exchanges'] = recent_exchanges
                    
                    # Process query with orchestrator
                    tool_result = asyncio.run(
                        asyncio.wait_for(
                            self.orchestrator.process_query(question, context),
                            timeout=15.0
                        )
                    )
                    
                    # Show tool execution status
                    if tool_result.get('execution_type') == 'chained':
                        print(f"\n[Using chained execution with {tool_result.get('chain_steps', 0)} steps]")
                        # Show each step as it executes
                        for step_num in range(1, tool_result.get('chain_steps', 0) + 1):
                            print(f"  Step {step_num}: Processing...")
                    elif tool_result.get('tools_executed'):
                        tools_str = ', '.join(tool_result['tools_executed'])
                        print(f"\n[Executing tools: {tools_str}]")
                    elif tool_result.get('tools_skipped'):
                        logger.debug(f"Tools skipped: {tool_result.get('skip_reason', 'Unknown')}")
                    
                    # Enhance question with tool results if available
                    if tool_result.get('results'):
                        # Show that we got results
                        total_tools = len(tool_result.get('tools_executed', []))
                        if tool_result.get('execution_type') == 'chained':
                            print(f"[Retrieved context from {total_tools} tool(s) across {tool_result.get('chain_steps', 1)} steps]")
                        else:
                            print(f"[Retrieved context from {len(tool_result['results'])} tool(s)]")
                        question = self._enhance_with_tool_context(question, tool_result['results'])
                    
                except Exception as e:
                    logger.warning(f"Tool execution failed: {e}")
                    # Continue without tools
            
            # Prepare context with memory
            if use_memory and self.current_memory:
                memory_context = self.current_memory.get_formatted_context()
                full_context = f"{self.system_prompt}\n\n{memory_context}" if memory_context else self.system_prompt
            else:
                full_context = self.system_prompt
            
            # Get response from LLM
            response = self.llm_client.complete(
                prompt=question,
                system_prompt=full_context,
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response.content
            
            # Update memory if in session
            if use_memory and self.current_memory:
                self.current_memory.add_exchange(question, answer)
                
                if self.current_session:
                    self.session_manager.update_session_stats(
                        message_count_delta=2,
                        token_count_delta=response.total_tokens,
                        cost_delta=response.cost
                    )
            
            # Log token usage if enabled
            if self.llm_client.config.log_tokens:
                logger.info(f"Tokens used - Input: {response.prompt_tokens}, Output: {response.completion_tokens}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in copilot ask: {e}")
            return self.persona.format_error_response(e)
    
    def _enhance_with_tool_context(self, question: str, tool_results: List[Dict[str, Any]]) -> str:
        """Enhance question with context from tool execution."""
        enhanced = question + "\n\n### ACTUAL PROJECT CONTENT (READ FROM FILES - ONLY DESCRIBE WHAT'S SHOWN BELOW):\n"
        enhanced += "IMPORTANT: Only reference files and content that appear in this context. Do not imagine or assume other files exist.\n"
        
        for result in tool_results:
            if result['status'] == 'success' and result.get('data'):
                tool_name = result['tool']
                data = result['data']
                
                if tool_name == 'read_file':
                    # Handle both string and dict data formats
                    if isinstance(data, str):
                        content = data
                        path = "file"
                    else:
                        content = data.get('content', data) if isinstance(data, dict) else str(data)
                        path = data.get('path', 'unknown') if isinstance(data, dict) else "file"
                    
                    enhanced += f"\n==== FILE CONTENT: {path} ====\n"
                    # Include more content for implementation understanding
                    enhanced += str(content)[:3000] + "\n"
                    if len(str(content)) > 3000:
                        enhanced += "... [truncated]\n"
                
                elif tool_name == 'list_directory' and data:
                    enhanced += f"\n==== COMPLETE DIRECTORY LISTING: {data.get('path', '.')} ====\n"
                    enhanced += "THE FOLLOWING IS THE COMPLETE LIST - NO OTHER FILES EXIST IN THIS DIRECTORY:\n"
                    files = data.get('files', [])
                    dirs = data.get('directories', [])
                    if dirs:
                        enhanced += f"Subdirectories ({len(dirs)} total): {', '.join(dirs)}\n"
                    else:
                        enhanced += "Subdirectories: NONE\n"
                    if files:
                        enhanced += f"Files ({len(files)} total): {', '.join(files)}\n"
                    else:
                        enhanced += "Files: NONE\n"
                    enhanced += "==== END OF LISTING ====\n"
                
                elif tool_name == 'search_in_files' and data:
                    matches = data.get('matches', data.get('results', []))
                    enhanced += f"\n==== SEARCH RESULTS: Found {len(matches)} matches ====\n"
                    for match in matches[:10]:
                        enhanced += f"  - {match}\n"
                
                elif tool_name == 'find_files' and data:
                    matches = data.get('matches', data.get('results', []))
                    enhanced += f"\n==== FILES FOUND: {len(matches)} matches ====\n"
                    for match in matches[:10]:
                        enhanced += f"  - {match}\n"
        
        return enhanced
    
    def chat(self, resume_session: Optional[str] = None):
        """
        Start an interactive chat session.
        
        Args:
            resume_session: Optional session ID to resume
        """
        # Create or resume session
        if resume_session:
            session = self.session_manager.recover_session(resume_session)
            if not session:
                print(f"Could not resume session: {resume_session}")
                return
            print(f"\nResumed session: {resume_session}")
        else:
            session = self.session_manager.create_session(
                project_path=str(self.project_path),
                persona_type=self.persona.name
            )
            print(f"\nStarted new session: {session.session_id}")
        
        self.current_session = session
        
        # Create memory for this session
        self.current_memory = self.memory_manager.create_memory(
            session_id=session.session_id,
            max_tokens=6000
        )
        
        # Load previous conversation if resuming
        if resume_session:
            session_path = self.session_manager.get_session_path(resume_session)
            memory_file = session_path / "memory.json"
            if memory_file.exists():
                self.current_memory.load_session(memory_file)
        
        print("\n" + "="*60)
        print(f"  Copilot Chat - Analyzing: {self.project_path.name}")
        print(f"  Session: {session.session_id}")
        print(f"  Persona: {self.persona.name}")
        print("="*60)
        print(f"\n{self.persona.get_greeting()}")
        print("\nType your questions below. Press Ctrl+C to exit.")
        print("Commands: /stats, /save, /clear, /help\n")
        
        try:
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        self._handle_command(user_input)
                        continue
                    
                    # Check for exit
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("\nCopilot: Goodbye! Feel free to return if you have more questions.")
                        break
                    
                    # Get and display response
                    print("\nCopilot: ", end="", flush=True)
                    response = self.ask(user_input, use_memory=True, use_tools=True)
                    
                    # Print response line by line
                    for line in response.split('\n'):
                        print(line)
                    
                except KeyboardInterrupt:
                    print("\n\nCopilot: Chat session ended. Saving session...")
                    break
                except EOFError:
                    print("\n\nCopilot: Chat session ended. Saving session...")
                    break
                except Exception as e:
                    print(f"\nCopilot: {self.persona.format_error_response(e)}")
        
        except Exception as e:
            logger.error(f"Chat session error: {e}")
            print(f"\nAn error occurred: {e}")
        
        finally:
            # Save session
            if self.current_memory and self.current_session:
                try:
                    memory_path = self.session_manager.get_session_path() / "memory.json"
                    self.current_memory.save_session(memory_path)
                    self.session_manager.end_session(SessionStatus.COMPLETED)
                    print(f"\nSession saved: {self.current_session.session_id}")
                    
                    stats = self.current_memory.get_stats()
                    print(f"Total messages: {stats['total_messages']}")
                    print(f"Files referenced: {stats['files_referenced']}")
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")
            
            # Clear current session
            self.current_memory = None
            self.current_session = None
    
    def _handle_command(self, command: str):
        """Handle chat commands."""
        cmd = command.lower().strip()
        
        if cmd == '/stats':
            if self.current_memory:
                stats = self.current_memory.get_stats()
                print("\n=== Session Statistics ===")
                print(f"Session ID: {stats['session_id']}")
                print(f"Messages: {stats['total_messages']}")
                print(f"Files referenced: {stats['files_referenced']}")
                print(f"Key facts: {stats['key_facts_count']}")
        
        elif cmd == '/save':
            if self.current_memory:
                path = self.session_manager.get_session_path() / "memory.json"
                self.current_memory.save_session(path)
                print(f"Session saved to: {path}")
        
        elif cmd == '/clear':
            if self.current_memory:
                self.current_memory.clear()
                print("Memory cleared for this session")
        
        elif cmd == '/help':
            print("\n=== Available Commands ===")
            print("/stats  - Show session statistics")
            print("/save   - Save current session")
            print("/clear  - Clear memory (start fresh)")
            print("/help   - Show this help message")
            print("\nType 'exit', 'quit', or 'bye' to end the session")
        
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get information about the project being analyzed."""
        return {
            "path": str(self.project_path),
            "name": self.project_path.name,
            "exists": self.project_path.exists(),
            "is_directory": self.project_path.is_dir(),
            "persona": self.persona.name,
            "llm_model": self.llm_client.config.model,
            "llm_provider": self.llm_client.config.provider.value,
            "has_project_context": bool(self.project_context),
            "mcp_enabled": self.use_mcp
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self.llm_client.get_stats()
        stats["project_path"] = str(self.project_path)
        
        if self.session_manager:
            stats["sessions"] = self.session_manager.get_session_summary()
        
        if self.current_memory:
            stats["current_session"] = self.current_memory.get_stats()
        
        return stats