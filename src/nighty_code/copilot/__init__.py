"""
Copilot module for AI-assisted code analysis and interaction.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from ..llm.client import LLMClient
from ..llm.config import LLMConfig, LLMProvider, TokenStrategy
from .persona import CopilotPersona, get_persona
from .memory import CopilotMemory, MemoryManager
from .session import SessionManager, SessionStatus

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CopilotClient:
    """
    Main copilot client for AI-assisted code analysis.
    
    This client provides an intelligent interface for understanding
    and navigating codebases using LLM capabilities.
    """
    
    def __init__(
        self,
        folder_path: Optional[str] = None,
        persona_type: str = "default",
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize the copilot client.
        
        Args:
            folder_path: Path to the project folder to analyze. Uses current directory if None.
            persona_type: Type of persona to use (default, architect, security, performance)
            llm_config: Optional LLM configuration. Uses environment defaults if None.
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
        
        # Initialize project understanding (Phase 1 & 2)
        self.project_context = None
        self._analyze_project()
        
        # Initialize LLM client
        if llm_config is None:
            llm_config = self._create_default_config()
        
        self.llm_client = LLMClient(llm_config)
        
        # System prompt will be generated after project analysis
        self.system_prompt = None
        
        # Memory and session management
        self.memory_manager = MemoryManager(self.llm_client)
        self.session_manager = SessionManager()
        self.current_memory: Optional[CopilotMemory] = None
        self.current_session = None
        
        # Generate system prompt with project context
        self._generate_system_prompt()
        
        logger.info(f"CopilotClient initialized with persona: {self.persona.name}")
    
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
        
        # Get model and other settings
        model = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022" if provider == LLMProvider.ANTHROPIC else "gpt-3.5-turbo")
        
        # Token strategies
        input_strategy = TokenStrategy.TRUNCATE if os.getenv("LLM_INPUT_STRATEGY", "truncate") == "truncate" else TokenStrategy.ERROR
        output_strategy = TokenStrategy.CONTINUE if os.getenv("LLM_OUTPUT_STRATEGY", "continue") == "continue" else TokenStrategy.ERROR
        
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            input_token_strategy=input_strategy,
            output_token_strategy=output_strategy,
            track_costs=os.getenv("LLM_TRACK_COSTS", "true").lower() == "true",
            cost_warning_threshold=float(os.getenv("LLM_COST_WARNING_THRESHOLD", "1.0")),
            log_requests=os.getenv("LLM_LOG_REQUESTS", "false").lower() == "true",
            log_responses=os.getenv("LLM_LOG_RESPONSES", "false").lower() == "true",
            log_tokens=os.getenv("LLM_LOG_TOKENS", "true").lower() == "true"
        )
    
    def _analyze_project(self):
        """Analyze project structure using progressive exploration."""
        try:
            print("Analyzing project structure...")
            
            # Use simplified progressive exploration
            exploration_result = self._progressive_explore(self.project_path, token_budget=2000)
            
            print(f"  Found {exploration_result['file_count']} files, {exploration_result['dir_count']} directories")
            print(f"  Used {exploration_result['tokens_used']} tokens")
            
            # Store the context directly
            if exploration_result['content']:
                # Join all content pieces
                self.project_context = "\n\n".join(exploration_result['content'])
                print("  Project context loaded")
            else:
                # Fallback to basic context
                self.project_context = f"Project: {self.project_path.name}\nLocation: {self.project_path}"
                print("  Using basic fallback context")
            
            print("  Project analysis complete!")
            
        except Exception as e:
            logger.error(f"Failed to analyze project: {e}")
            # Fallback to basic context
            self.project_context = f"Project: {self.project_path.name}\nLocation: {self.project_path}"
    
    def _progressive_explore(self, path: Path, token_budget: int = 4000) -> Dict:
        """
        Progressive folder exploration based on token budget.
        
        Args:
            path: Path to explore
            token_budget: Maximum tokens to use for exploration
            
        Returns:
            Dictionary with exploration results
        """
        result = {
            'content': [],
            'file_count': 0,
            'dir_count': 0,
            'tokens_used': 0
        }
        
        def estimate_tokens(text: str) -> int:
            """Rough token estimation (1 token = ~4 chars)"""
            return len(text) // 4
        
        def add_content_if_budget_allows(content: str) -> bool:
            """Add content if token budget allows, return success"""
            tokens_needed = estimate_tokens(content)
            if result['tokens_used'] + tokens_needed <= token_budget:
                result['content'].append(content)
                result['tokens_used'] += tokens_needed
                return True
            return False
        
        try:
            # Collect items
            items = []
            for item in path.iterdir():
                if item.name.startswith('.'):
                    continue  # Skip hidden files/dirs
                items.append(item)
                if item.is_dir():
                    result['dir_count'] += 1
                else:
                    result['file_count'] += 1
            
            # Sort by priority (README first, then important dirs, then files)
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
            priority_files = ['README.md', 'README.txt', 'README', 'package.json', 'requirements.txt', 'setup.py', 'pyproject.toml']
            
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
                
                # Directory listings for key folders
                elif item.is_dir() and item.name in ['src', 'lib', 'app', 'core']:
                    try:
                        subfiles = []
                        subdirs = []
                        for subitem in item.iterdir():
                            if subitem.name.startswith('.'):
                                continue
                            if subitem.is_dir():
                                subdirs.append(subitem.name)
                            else:
                                subfiles.append(subitem.name)
                        
                        dir_desc = f"\n--- {item.name}/ ---\n"
                        if subdirs:
                            dir_desc += f"Subdirectories: {', '.join(subdirs[:8])}\n"
                        if subfiles:
                            dir_desc += f"Files: {', '.join(subfiles[:8])}\n"
                        
                        add_content_if_budget_allows(dir_desc)
                    except Exception:
                        continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error in progressive exploration: {e}")
            result['content'] = [f"Project: {path.name}\nLocation: {path}\nExploration failed: {str(e)}"]
            result['tokens_used'] = estimate_tokens(result['content'][0])
            return result
    
    def _generate_system_prompt(self):
        """Generate system prompt with project context."""
        base_prompt = self.persona.generate_system_prompt(str(self.project_path))
        
        if self.project_context:
            self.system_prompt = f"""{base_prompt}

## PROJECT CONTEXT

{self.project_context}

---

You have comprehensive knowledge about this project's structure, technologies, and purpose. 
Use this context to provide accurate, specific assistance."""
        else:
            self.system_prompt = base_prompt
    
    def ask(self, question: str, use_memory: bool = True) -> str:
        """
        Ask the copilot a question about the codebase.
        
        Args:
            question: The question to ask
            use_memory: Whether to use conversation memory
            
        Returns:
            The copilot's response
        """
        try:
            # Prepare context with memory if available
            if use_memory and self.current_memory:
                # Get memory context
                memory_context = self.current_memory.get_formatted_context()
                
                # Combine system prompt with memory
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
            
            # Update memory if in a session
            if use_memory and self.current_memory:
                self.current_memory.add_exchange(question, answer)
                
                # Update session stats
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
    
    def chat(self, resume_session: Optional[str] = None):
        """
        Start an interactive chat session in the terminal.
        
        This method provides a simple terminal interface for chatting
        with the copilot about the codebase. Each chat() call creates
        a new session with its own memory. Exit with Ctrl+C.
        
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
        
        # Create memory for this session (reduced to leave room for project context)
        self.current_memory = self.memory_manager.create_memory(
            session_id=session.session_id,
            max_tokens=6000  # 6000 for conversation + 2000 for project context = 8000 total
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
                    # Get user input
                    user_input = input("\nYou: ").strip()
                    
                    # Skip empty input
                    if not user_input:
                        continue
                    
                    # Check for commands
                    if user_input.startswith('/'):
                        self._handle_command(user_input)
                        continue
                    
                    # Check for exit commands
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("\nCopilot: Goodbye! Feel free to return if you have more questions.")
                        break
                    
                    # Get and display response
                    print("\nCopilot: ", end="", flush=True)
                    response = self.ask(user_input, use_memory=True)
                    
                    # Print response with nice formatting
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
            # Save session before exiting
            if self.current_memory and self.current_session:
                try:
                    # Save memory
                    memory_path = self.session_manager.get_session_path() / "memory.json"
                    self.current_memory.save_session(memory_path)
                    
                    # End session
                    self.session_manager.end_session(SessionStatus.COMPLETED)
                    print(f"\nSession saved: {self.current_session.session_id}")
                    
                    # Show stats
                    stats = self.current_memory.get_stats()
                    print(f"Total messages: {stats['total_messages']}")
                    print(f"Files referenced: {stats['files_referenced']}")
                    
                except Exception as e:
                    logger.error(f"Failed to save session: {e}")
            
            # Clear current session and memory
            self.current_memory = None
            self.current_session = None
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get information about the project being analyzed.
        
        Returns:
            Dictionary with project information
        """
        info = {
            "path": str(self.project_path),
            "name": self.project_path.name,
            "exists": self.project_path.exists(),
            "is_directory": self.project_path.is_dir(),
            "persona": self.persona.name,
            "llm_model": self.llm_client.config.model,
            "llm_provider": self.llm_client.config.provider.value
        }
        
        # Add context availability
        if self.project_context:
            info["has_project_context"] = True
            info["context_size"] = len(self.project_context)
        
        return info
    
    def _handle_command(self, command: str):
        """Handle special commands during chat."""
        cmd = command.lower().strip()
        
        if cmd == '/stats':
            if self.current_memory:
                stats = self.current_memory.get_stats()
                print("\n=== Session Statistics ===")
                print(f"Session ID: {stats['session_id']}")
                print(f"Messages: {stats['total_messages']}")
                print(f"Files referenced: {stats['files_referenced']}")
                print(f"Key facts: {stats['key_facts_count']}")
                print(f"Checkpoints: {stats['checkpoints']}")
        
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
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent chat sessions."""
        sessions = self.session_manager.list_sessions(
            project_path=str(self.project_path),
            limit=limit
        )
        return [s.to_dict() for s in sessions]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = self.llm_client.get_stats()
        stats["project_path"] = str(self.project_path)
        
        # Add session stats if available
        if self.session_manager:
            session_summary = self.session_manager.get_session_summary()
            stats["sessions"] = session_summary
        
        # Add current memory stats if in session
        if self.current_memory:
            stats["current_session"] = self.current_memory.get_stats()
        
        return stats


# Export main classes
__all__ = ["CopilotClient"]