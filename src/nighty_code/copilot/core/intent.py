"""
Unified intent recognition system.
Combines pattern matching with LLM fallback for ambiguous queries.
"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""
    FILE_READ = "file_read"
    FILE_SEARCH = "file_search"
    DIR_LIST = "dir_list"
    CODE_SEARCH = "code_search"
    EXPLAIN = "explain"
    EXPLORE = "explore"
    GENERAL = "general"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    FILE_PATH = "file_path"
    FILE_NAME = "file_name"
    DIR_PATH = "dir_path"
    CODE_PATTERN = "code_pattern"
    CLASS_NAME = "class_name"
    FUNCTION_NAME = "function_name"
    VARIABLE = "variable"


@dataclass
class Entity:
    """Extracted entity from query."""
    type: EntityType
    value: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedIntent:
    """Processed intent with all extracted information."""
    type: IntentType
    confidence: float
    entities: List[Entity]
    keywords: List[str]
    suggested_tools: List[str]
    reasoning: str
    ambiguous: bool = False


class IntentRecognizer:
    """
    Unified intent recognizer with pattern matching and optional LLM support.
    """
    
    def __init__(
        self,
        llm_client=None,
        project_root: Optional[Path] = None,
        enable_llm_fallback: bool = True
    ):
        """
        Initialize intent recognizer.
        
        Args:
            llm_client: Optional structured LLM client for ambiguous queries
            project_root: Project root path for context
            enable_llm_fallback: Whether to use LLM for ambiguous queries
        """
        self.llm_client = llm_client
        self.project_root = project_root or Path.cwd()
        self.enable_llm_fallback = enable_llm_fallback and llm_client is not None
        
        # Compile patterns for efficiency
        self.patterns = self._compile_patterns()
        
        # Keywords for intent detection
        self.keywords = {
            IntentType.FILE_READ: {'read', 'show', 'display', 'view', 'open', 'see', 'look'},
            IntentType.FILE_SEARCH: {'find', 'search', 'locate', 'where'},
            IntentType.DIR_LIST: {'list', 'ls', 'dir', 'browse', 'explore', 'files'},
            IntentType.CODE_SEARCH: {'grep', 'search', 'find', 'pattern', 'regex'},
            IntentType.EXPLAIN: {'explain', 'what', 'how', 'why', 'describe', 'tell'},
            IntentType.EXPLORE: {'explore', 'structure', 'overview', 'architecture'},
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for entity extraction."""
        return {
            'file_path': re.compile(r'[\w\-/\\]+\.\w{2,4}'),
            'quoted': re.compile(r'["\']([^"\']+)["\']'),
            'directory': re.compile(r'\b(?:in|from|at|under)\s+([\w\-/\\]+)/?'),
            'class_def': re.compile(r'\b(?:class|interface|struct)\s+(\w+)'),
            'function_def': re.compile(r'\b(?:function|def|method|func)\s+(\w+)'),
            'code_pattern': re.compile(r'(?:TODO|FIXME|BUG|HACK|XXX)'),
        }
    
    async def recognize(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> ProcessedIntent:
        """
        Recognize intent from user query.
        
        Args:
            query: User's query
            context: Optional context (current file, directory, etc.)
            request_id: Optional request ID for tracing
            
        Returns:
            ProcessedIntent with recognized intent and entities
        """
        if not query:
            return self._unknown_intent("Empty query")
        
        query_lower = query.lower()
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Calculate intent scores
        intent_scores = self._calculate_intent_scores(query_lower, keywords, entities)
        
        # Check for obvious patterns first
        if obvious_intent := self._check_obvious_patterns(query_lower, entities):
            return ProcessedIntent(
                type=obvious_intent,
                confidence=0.9,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools(obvious_intent, entities),
                reasoning="Clear pattern match",
                ambiguous=False
            )
        
        # If no clear pattern and LLM is available, use it
        if self.enable_llm_fallback and not intent_scores:
            return await self._llm_interpret(query, entities, keywords, context)
        
        # Use best scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_type, confidence = best_intent
            
            return ProcessedIntent(
                type=intent_type,
                confidence=confidence,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools(intent_type, entities),
                reasoning="Keyword and entity analysis",
                ambiguous=confidence < 0.7
            )
        
        # Default to unknown
        return self._unknown_intent("No clear intent recognized")
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities from query."""
        entities = []
        
        # Extract file paths
        for match in self.patterns['file_path'].finditer(query):
            entities.append(Entity(
                type=EntityType.FILE_PATH,
                value=match.group(0),
                confidence=0.9
            ))
        
        # Extract quoted strings
        for match in self.patterns['quoted'].finditer(query):
            value = match.group(1)
            # Determine entity type based on content
            if '.' in value and value.count('.') == 1:
                entity_type = EntityType.FILE_NAME
            elif '/' in value or '\\' in value:
                entity_type = EntityType.DIR_PATH
            else:
                entity_type = EntityType.CODE_PATTERN
            
            entities.append(Entity(
                type=entity_type,
                value=value,
                confidence=0.8
            ))
        
        # Extract directories
        for match in self.patterns['directory'].finditer(query):
            entities.append(Entity(
                type=EntityType.DIR_PATH,
                value=match.group(1),
                confidence=0.7
            ))
        
        # Extract code elements
        for match in self.patterns['class_def'].finditer(query):
            entities.append(Entity(
                type=EntityType.CLASS_NAME,
                value=match.group(1),
                confidence=0.8
            ))
        
        for match in self.patterns['function_def'].finditer(query):
            entities.append(Entity(
                type=EntityType.FUNCTION_NAME,
                value=match.group(1),
                confidence=0.8
            ))
        
        return entities
    
    def _extract_keywords(self, query_lower: str) -> List[str]:
        """Extract relevant keywords from query."""
        # Simple word extraction
        words = query_lower.split()
        
        # Filter to meaningful words
        keywords = []
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w]', '', word)
            
            # Keep if it's a known keyword or looks meaningful
            if len(word) > 2:
                for intent_keywords in self.keywords.values():
                    if word in intent_keywords:
                        keywords.append(word)
                        break
        
        return keywords
    
    def _calculate_intent_scores(
        self,
        query_lower: str,
        keywords: List[str],
        entities: List[Entity]
    ) -> Dict[IntentType, float]:
        """Calculate confidence scores for each intent type."""
        scores = {}
        
        for intent_type, intent_keywords in self.keywords.items():
            score = 0.0
            
            # Keyword matching
            for keyword in keywords:
                if keyword in intent_keywords:
                    score += 0.3
            
            # Direct word matching
            words = set(query_lower.split())
            matching_words = words & intent_keywords
            if matching_words:
                score += 0.2 * len(matching_words)
            
            # Entity boost
            if self._entities_match_intent(entities, intent_type):
                score += 0.3
            
            if score > 0:
                scores[intent_type] = min(score, 0.95)  # Cap at 0.95
        
        return scores
    
    def _check_obvious_patterns(
        self,
        query_lower: str,
        entities: List[Entity]
    ) -> Optional[IntentType]:
        """Check for obvious intent patterns."""
        # Direct file read patterns
        if any(phrase in query_lower for phrase in ['show me', 'read', 'open', 'view']):
            if any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities):
                return IntentType.FILE_READ
        
        # Directory listing
        if any(phrase in query_lower for phrase in ['list files', 'ls ', 'dir ']):
            return IntentType.DIR_LIST
        
        # Code search
        if 'grep' in query_lower or ('search' in query_lower and 'for' in query_lower):
            return IntentType.CODE_SEARCH
        
        # Explanation
        if query_lower.startswith(('what', 'how', 'why', 'explain')):
            return IntentType.EXPLAIN
        
        return None
    
    def _entities_match_intent(self, entities: List[Entity], intent_type: IntentType) -> bool:
        """Check if entities match the intent type."""
        if intent_type == IntentType.FILE_READ:
            return any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities)
        elif intent_type == IntentType.DIR_LIST:
            return any(e.type == EntityType.DIR_PATH for e in entities)
        elif intent_type == IntentType.CODE_SEARCH:
            return any(e.type == EntityType.CODE_PATTERN for e in entities)
        elif intent_type == IntentType.EXPLAIN:
            return any(e.type in [EntityType.CLASS_NAME, EntityType.FUNCTION_NAME] for e in entities)
        return False
    
    async def _llm_interpret(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
        context: Optional[Dict[str, Any]]
    ) -> ProcessedIntent:
        """Use LLM to interpret ambiguous queries."""
        if not self.llm_client:
            return self._unknown_intent("LLM not available")
        
        # Build prompt
        prompt = f"""Interpret this user query for a code assistant tool.

Query: "{query}"

Context:
- Keywords found: {', '.join(keywords) if keywords else 'none'}
- Entities detected: {', '.join(f"{e.type.value}:{e.value}" for e in entities) if entities else 'none'}

Determine the most likely intent from:
- FILE_READ: User wants to read/view a specific file
- FILE_SEARCH: User wants to find files
- DIR_LIST: User wants to list directory contents
- CODE_SEARCH: User wants to search for code patterns
- EXPLAIN: User wants explanation about code
- EXPLORE: User wants to browse/explore the project
- GENERAL: General question about the project
- UNKNOWN: Query doesn't match any clear intent

Respond with the intent, reasoning, and whether it's ambiguous."""

        try:
            # Define response model
            class IntentResponse(BaseModel):
                intent: str
                reasoning: str
                ambiguous: bool
            
            # Get LLM response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.llm_client.generate,
                prompt,
                IntentResponse
            )
            
            # Parse response
            intent_str = response.intent.upper()
            intent_type = IntentType[intent_str] if intent_str in IntentType.__members__ else IntentType.UNKNOWN
            
            confidence = 0.7 if intent_type != IntentType.UNKNOWN else 0.3
            if response.ambiguous:
                confidence *= 0.8
            
            return ProcessedIntent(
                type=intent_type,
                confidence=confidence,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools(intent_type, entities),
                reasoning=response.reasoning,
                ambiguous=response.ambiguous
            )
            
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            return self._unknown_intent(f"LLM interpretation failed: {str(e)[:50]}")
    
    def _suggest_tools(self, intent_type: IntentType, entities: List[Entity]) -> List[str]:
        """Suggest tools based on intent and entities."""
        tool_mapping = {
            IntentType.FILE_READ: ['read_file'],
            IntentType.FILE_SEARCH: ['search_files', 'fuzzy_find'],
            IntentType.DIR_LIST: ['list_directory'],
            IntentType.CODE_SEARCH: ['search_pattern'],
            IntentType.EXPLAIN: ['read_file', 'search_pattern'],
            IntentType.EXPLORE: ['list_directory', 'search_files'],
            IntentType.GENERAL: ['search_files'],
            IntentType.UNKNOWN: ['search_files']
        }
        
        tools = tool_mapping.get(intent_type, ['search_files'])
        
        # Add fuzzy find if no specific file
        if intent_type == IntentType.FILE_READ:
            if not any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities):
                tools.insert(0, 'fuzzy_find')
        
        return tools
    
    def _unknown_intent(self, reason: str) -> ProcessedIntent:
        """Create unknown intent response."""
        return ProcessedIntent(
            type=IntentType.UNKNOWN,
            confidence=0.1,
            entities=[],
            keywords=[],
            suggested_tools=['search_files'],
            reasoning=reason,
            ambiguous=True
        )