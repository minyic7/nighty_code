"""
Advanced memory management for Copilot using LangChain.

This module leverages LangChain's battle-tested memory implementations
while adding custom features specific to code analysis conversations.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory.chat_memory import BaseChatMemory
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from ..llm.client import LLMClient
from ..llm.token_utils import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    total_messages: int = 0
    total_tokens: int = 0
    compression_count: int = 0
    current_buffer_size: int = 0
    summary_size: int = 0
    files_referenced: set = field(default_factory=set)
    key_decisions: list = field(default_factory=list)


class LangChainLLMAdapter:
    """
    Adapter to make our LLMClient compatible with LangChain's interface.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.token_counter = TokenCounter()
    
    def predict(self, text: str) -> str:
        """LangChain compatible prediction method."""
        response = self.llm_client.complete(
            prompt=text,
            temperature=0.7,
            max_tokens=500  # For summaries
        )
        return response.content
    
    def get_num_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count_tokens(text, self.llm_client.config.model)
    
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            total += self.get_num_tokens(msg.content)
        return total


class CopilotMemory:
    """
    Advanced memory management for Copilot conversations.
    
    Features:
    - Automatic summarization when token limit approached
    - Sliding window of recent messages
    - Key facts and decision tracking
    - File/folder reference tracking
    - Session persistence support
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_token_limit: int = 6000,  # Leave room for 2000-token project context
        moving_summary_buffer: int = 2000,
        return_messages: bool = True,
        session_id: Optional[str] = None
    ):
        """
        Initialize Copilot memory.
        
        Args:
            llm_client: LLM client for summarization
            max_token_limit: Maximum tokens to keep in memory
            moving_summary_buffer: Token buffer before triggering summarization
            return_messages: Whether to return messages or string
            session_id: Optional session ID for persistence
        """
        self.llm_client = llm_client
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create appropriate LangChain LLM based on provider
        langchain_llm = self._create_langchain_llm(llm_client)
        
        # Use simpler memory for now - ConversationBufferWindowMemory
        # This avoids the summarization issues while still providing memory
        self.base_memory = ConversationBufferWindowMemory(
            k=20,  # Keep last 20 exchanges
            return_messages=return_messages,
            input_key="input",
            output_key="output",
            memory_key="history"
        )
        
        # Store the LangChain LLM for potential summarization
        self.langchain_llm = langchain_llm
        self.max_token_limit = max_token_limit
        
        # Custom tracking
        self.stats = MemoryStats()
        self.key_facts: List[str] = []
        self.file_references: set = set()
        self.code_snippets: List[Dict[str, str]] = []
        self.important_messages: List[Tuple[str, str]] = []  # (role, content) pairs
        
        # Checkpoints for recovery
        self.checkpoints: List[Dict[str, Any]] = []
        self.last_checkpoint_size = 0
        
        logger.info(f"CopilotMemory initialized for session: {self.session_id}")
    
    def _create_langchain_llm(self, llm_client: LLMClient):
        """
        Create a LangChain-compatible LLM instance.
        
        Args:
            llm_client: Our LLMClient instance
            
        Returns:
            LangChain LLM instance
        """
        from ..llm.config import LLMProvider
        
        # Get configuration from our client
        config = llm_client.config
        
        try:
            if config.provider == LLMProvider.ANTHROPIC:
                # Create Anthropic LLM for LangChain
                return ChatAnthropic(
                    model=config.model,
                    anthropic_api_key=config.api_key,
                    temperature=0.7,
                    max_tokens=1000
                )
            elif config.provider == LLMProvider.OPENAI:
                # Create OpenAI LLM for LangChain
                return ChatOpenAI(
                    model=config.model,
                    openai_api_key=config.api_key,
                    temperature=0.7,
                    max_tokens=1000
                )
            else:
                # Fallback to a simple wrapper if provider not directly supported
                logger.warning(f"Provider {config.provider} not directly supported by LangChain, using adapter")
                return LangChainLLMAdapter(llm_client)
        except Exception as e:
            logger.warning(f"Failed to create native LangChain LLM: {e}, using adapter")
            return LangChainLLMAdapter(llm_client)
    
    def add_exchange(self, user_input: str, assistant_response: str):
        """
        Add a conversation exchange to memory.
        
        Args:
            user_input: User's message
            assistant_response: Assistant's response
        """
        # Update LangChain memory
        self.base_memory.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
        
        # Update statistics
        self.stats.total_messages += 2
        self.stats.current_buffer_size = len(self.base_memory.buffer)
        
        # Extract and track important information
        self._extract_references(user_input, assistant_response)
        self._extract_key_facts(user_input, assistant_response)
        
        # Check if checkpoint needed
        if self.stats.total_messages - self.last_checkpoint_size >= 10:
            self._create_checkpoint()
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get memory variables for LLM context.
        
        Returns:
            Dictionary with memory context
        """
        # Get base memory
        base_vars = self.base_memory.load_memory_variables({})
        
        # Enhance with custom context
        enhanced_context = {
            "history": base_vars.get("history", []),
            "key_facts": self.key_facts[-5:] if self.key_facts else [],
            "files_discussed": list(self.file_references)[-10:] if self.file_references else [],
            "session_id": self.session_id
        }
        
        return enhanced_context
    
    def get_formatted_context(self) -> str:
        """
        Get formatted context string for LLM.
        
        Returns:
            Formatted context string
        """
        memory_vars = self.get_memory_variables()
        
        context_parts = []
        
        # Add conversation history
        if memory_vars["history"]:
            if isinstance(memory_vars["history"], list):
                # Format messages
                history_str = "\n".join([
                    f"{msg.type}: {msg.content}" 
                    for msg in memory_vars["history"]
                ])
                context_parts.append(f"Conversation History:\n{history_str}")
            else:
                context_parts.append(f"Conversation Summary:\n{memory_vars['history']}")
        
        # Add key facts if any
        if memory_vars["key_facts"]:
            facts_str = "\n".join(f"- {fact}" for fact in memory_vars["key_facts"])
            context_parts.append(f"\nKey Facts:\n{facts_str}")
        
        # Add file references if any
        if memory_vars["files_discussed"]:
            files_str = ", ".join(memory_vars["files_discussed"])
            context_parts.append(f"\nFiles Discussed: {files_str}")
        
        return "\n\n".join(context_parts)
    
    def _extract_references(self, user_input: str, assistant_response: str):
        """Extract file and folder references from messages."""
        import re
        
        # Pattern for file paths
        file_pattern = r'[\w/\\]+\.\w+|src/[\w/]+|tests/[\w/]+'
        
        for text in [user_input, assistant_response]:
            matches = re.findall(file_pattern, text)
            for match in matches:
                self.file_references.add(match)
                self.stats.files_referenced.add(match)
    
    def _extract_key_facts(self, user_input: str, assistant_response: str):
        """Extract key facts and decisions from conversation."""
        # Simple heuristics for now - can be enhanced with NLP
        keywords = ["decided", "conclusion", "important", "note that", "remember"]
        
        for text in [assistant_response]:  # Focus on assistant's statements
            lower_text = text.lower()
            for keyword in keywords:
                if keyword in lower_text:
                    # Extract sentence containing keyword
                    sentences = text.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            fact = sentence.strip()
                            if fact and len(fact) > 10:  # Avoid trivial facts
                                self.key_facts.append(fact)
                                self.stats.key_decisions.append(fact)
                            break
    
    def _create_checkpoint(self):
        """Create a checkpoint for recovery."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "message_count": self.stats.total_messages,
            "summary": "",  # ConversationBufferWindowMemory doesn't have summary
            "key_facts": self.key_facts.copy(),
            "file_references": list(self.file_references)
        }
        
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_size = self.stats.total_messages
        
        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)
    
    def clear(self):
        """Clear all memory."""
        self.base_memory.clear()
        self.stats = MemoryStats()
        self.key_facts.clear()
        self.file_references.clear()
        self.code_snippets.clear()
        self.important_messages.clear()
        self.checkpoints.clear()
        logger.info(f"Memory cleared for session: {self.session_id}")
    
    def save_session(self, filepath: Optional[Path] = None) -> Path:
        """
        Save session to file.
        
        Args:
            filepath: Optional path to save to
            
        Returns:
            Path where session was saved
        """
        if filepath is None:
            filepath = Path(f".copilot/sessions/{self.session_id}.json")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "total_messages": self.stats.total_messages,
                "total_tokens": self.stats.total_tokens,
                "compression_count": self.stats.compression_count
            },
            "key_facts": self.key_facts,
            "file_references": list(self.file_references),
            "checkpoints": self.checkpoints,
            "memory_buffer": [
                {"type": msg.type, "content": msg.content}
                for msg in self.base_memory.buffer
            ] if hasattr(self.base_memory, 'buffer') else []
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session saved to: {filepath}")
        return filepath
    
    def load_session(self, filepath: Path) -> bool:
        """
        Load session from file.
        
        Args:
            filepath: Path to session file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            self.session_id = session_data["session_id"]
            self.key_facts = session_data.get("key_facts", [])
            self.file_references = set(session_data.get("file_references", []))
            self.checkpoints = session_data.get("checkpoints", [])
            
            # Restore stats
            stats = session_data.get("stats", {})
            self.stats.total_messages = stats.get("total_messages", 0)
            self.stats.total_tokens = stats.get("total_tokens", 0)
            
            # Restore memory buffer if available
            # For ConversationBufferWindowMemory, we need to reconstruct the conversation
            for msg_data in session_data.get("memory_buffer", []):
                # Use save_context to properly add messages to memory
                if msg_data["type"] == "human":
                    # Find the next AI message
                    ai_response = None
                    buffer = session_data.get("memory_buffer", [])
                    current_idx = buffer.index(msg_data)
                    if current_idx + 1 < len(buffer) and buffer[current_idx + 1]["type"] == "ai":
                        ai_response = buffer[current_idx + 1]["content"]
                    
                    if ai_response:
                        self.base_memory.save_context(
                            {"input": msg_data["content"]},
                            {"output": ai_response}
                        )
            
            logger.info(f"Session loaded from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "total_messages": self.stats.total_messages,
            "current_buffer_size": self.stats.current_buffer_size,
            "files_referenced": len(self.stats.files_referenced),
            "key_facts_count": len(self.key_facts),
            "checkpoints": len(self.checkpoints),
            "compression_count": self.stats.compression_count
        }


class MemoryManager:
    """
    Manages multiple memory instances and strategies.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.active_memories: Dict[str, CopilotMemory] = {}
    
    def create_memory(
        self,
        session_id: Optional[str] = None,
        max_tokens: int = 8000
    ) -> CopilotMemory:
        """Create a new memory instance."""
        memory = CopilotMemory(
            llm_client=self.llm_client,
            max_token_limit=max_tokens,
            session_id=session_id
        )
        
        if session_id:
            self.active_memories[session_id] = memory
        
        return memory
    
    def get_memory(self, session_id: str) -> Optional[CopilotMemory]:
        """Get memory by session ID."""
        return self.active_memories.get(session_id)
    
    def cleanup_old_sessions(self, keep_last: int = 5):
        """Clean up old sessions, keeping only the most recent."""
        if len(self.active_memories) > keep_last:
            # Sort by session ID (timestamp-based)
            sorted_sessions = sorted(self.active_memories.keys())
            
            # Remove oldest sessions
            for session_id in sorted_sessions[:-keep_last]:
                del self.active_memories[session_id]
                logger.info(f"Cleaned up session: {session_id}")