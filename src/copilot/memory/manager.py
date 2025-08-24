# src/copilot/memory/manager.py
"""
Memory management for the copilot
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..core.types import Memory, CopilotState

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages short-term and long-term memory for the copilot"""
    
    def __init__(self, persistence_file: Optional[str] = None):
        self.persistence_file = persistence_file
        self.short_term_memory: List[Memory] = []
        self.long_term_memory: List[Memory] = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize memory manager"""
        if self._initialized:
            return
        
        # Load persisted memories if file exists
        if self.persistence_file:
            await self.load_memories()
        
        self._initialized = True
        logger.info("MemoryManager initialized")
    
    async def store_interaction(self, state: CopilotState):
        """Store the current interaction in memory"""
        # Get the last user message and response
        if len(state.messages) < 2:
            return
        
        last_user_msg = None
        last_response = None
        
        for msg in reversed(state.messages):
            if hasattr(msg, 'thoughts') and not last_response:
                last_response = msg
            elif not hasattr(msg, 'thoughts') and not last_user_msg:
                last_user_msg = msg
            
            if last_user_msg and last_response:
                break
        
        if last_user_msg and last_response:
            # Create memory of interaction
            memory = Memory(
                content=f"User asked: {last_user_msg.content}\nI responded: {last_response.content}",
                type="conversation",
                importance=0.7
            )
            state.add_memory(memory)
    
    async def recall(self, query: str, state: CopilotState, limit: int = 5) -> List[Memory]:
        """Recall relevant memories based on query"""
        relevant_memories = []
        
        # Simple relevance scoring based on content similarity
        # In production, this would use embeddings or more sophisticated retrieval
        all_memories = state.short_term_memory + state.long_term_memory
        
        scored_memories = []
        for memory in all_memories:
            score = self._calculate_relevance(query, memory.content)
            if score > 0.3:  # Threshold for relevance
                scored_memories.append((score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        relevant_memories = [m for _, m in scored_memories[:limit]]
        
        return relevant_memories
    
    async def consolidate_memories(self, state: CopilotState):
        """Consolidate short-term memories into long-term"""
        # Move important short-term memories to long-term
        important_threshold = 0.7
        
        important_memories = [
            m for m in state.short_term_memory 
            if m.importance >= important_threshold
        ]
        
        # Add to long-term memory
        for memory in important_memories:
            state.add_memory(memory, long_term=True)
        
        # Keep only recent short-term memories
        cutoff_time = datetime.now() - timedelta(hours=1)
        state.short_term_memory = [
            m for m in state.short_term_memory
            if m.timestamp > cutoff_time
        ]
    
    async def save_memories(self, state: CopilotState):
        """Save memories to persistence file"""
        if not self.persistence_file:
            return
        
        try:
            memories_data = {
                "short_term": [m.model_dump() for m in state.short_term_memory],
                "long_term": [m.model_dump() for m in state.long_term_memory],
                "saved_at": datetime.now().isoformat()
            }
            
            path = Path(self.persistence_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(memories_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(state.short_term_memory) + len(state.long_term_memory)} memories")
        
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    async def load_memories(self) -> tuple[List[Memory], List[Memory]]:
        """Load memories from persistence file"""
        if not self.persistence_file or not Path(self.persistence_file).exists():
            return [], []
        
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            short_term = [Memory(**m) for m in data.get("short_term", [])]
            long_term = [Memory(**m) for m in data.get("long_term", [])]
            
            logger.info(f"Loaded {len(short_term) + len(long_term)} memories")
            return short_term, long_term
        
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            return [], []
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        # Simple keyword-based scoring
        # In production, use embeddings or semantic similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0