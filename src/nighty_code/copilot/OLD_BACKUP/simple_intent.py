"""
Simple intent recognition that just works.
No overengineering, just the essentials.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class IntentType(Enum):
    """Basic intent types."""
    FILE_READ = "file_read"
    FILE_SEARCH = "file_search"
    DIR_LIST = "dir_list"
    CODE_SEARCH = "code_search"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Simple entity representation."""
    type: str  # 'file', 'dir', 'pattern', 'code'
    value: str


@dataclass
class Intent:
    """Simple intent representation."""
    type: IntentType
    entities: List[Entity]
    suggested_tools: List[str]


class SimpleIntentRecognizer:
    """
    Simple intent recognizer that just works.
    No caching, no metrics, no rate limiting - just recognition.
    """
    
    def __init__(self):
        # Pre-compile patterns for efficiency
        self.patterns = {
            'file': re.compile(r'\b([a-zA-Z0-9_\-./\\]+\.[a-zA-Z]{2,4})\b'),
            'quoted': re.compile(r'["\']([^"\']+)["\']'),
            'directory': re.compile(r'\b(?:in|from|at)\s+([a-zA-Z0-9_\-/\\]+)/?'),
            'code_element': re.compile(r'\b(class|function|def|method)\s+(\w+)'),
        }
        
        # Simple keyword mapping
        self.keywords = {
            IntentType.FILE_READ: {'read', 'show', 'display', 'view', 'open', 'see'},
            IntentType.FILE_SEARCH: {'find', 'search', 'locate', 'where'},
            IntentType.DIR_LIST: {'list', 'ls', 'dir', 'browse', 'explore'},
            IntentType.CODE_SEARCH: {'grep', 'search', 'find'},
        }
    
    def recognize(self, query: str) -> Intent:
        """
        Recognize intent from query.
        Simple, deterministic, testable.
        """
        if not query:
            return Intent(IntentType.UNKNOWN, [], [])
        
        query_lower = query.lower()
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine intent type (pass original query too for case-sensitive checks)
        intent_type = self._determine_intent(query_lower, entities, query)
        
        # Suggest tools
        tools = self._suggest_tools(intent_type, entities)
        
        return Intent(intent_type, entities, tools)
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities from query."""
        entities = []
        
        # Extract file names
        for match in self.patterns['file'].finditer(query):
            entities.append(Entity('file', match.group(1)))
        
        # Extract quoted strings
        for match in self.patterns['quoted'].finditer(query):
            value = match.group(1)
            # Determine if it's a pattern or literal
            entity_type = 'pattern' if '*' in value or '?' in value else 'literal'
            entities.append(Entity(entity_type, value))
        
        # Extract directories
        for match in self.patterns['directory'].finditer(query):
            entities.append(Entity('dir', match.group(1)))
        
        # Extract code elements
        for match in self.patterns['code_element'].finditer(query):
            entities.append(Entity('code', match.group(2)))
        
        return entities
    
    def _determine_intent(self, query_lower: str, entities: List[Entity], original_query: str) -> IntentType:
        """Determine intent from query and entities."""
        words = set(query_lower.split())
        
        # Check for specific patterns first
        if any(e.type == 'file' for e in entities):
            if words & self.keywords[IntentType.FILE_READ]:
                return IntentType.FILE_READ
            elif words & self.keywords[IntentType.FILE_SEARCH]:
                return IntentType.FILE_SEARCH
        
        if any(e.type == 'dir' for e in entities):
            if words & self.keywords[IntentType.DIR_LIST]:
                return IntentType.DIR_LIST
        
        # Check for code search patterns (even without entities)
        if words & self.keywords[IntentType.CODE_SEARCH]:
            # "search for X" should be code search if X looks like code
            if any(word.isupper() for word in original_query.split()):  # TODO, FIXME, etc
                return IntentType.CODE_SEARCH
            elif any(e.type in ['pattern', 'literal'] for e in entities):
                return IntentType.CODE_SEARCH
        
        # Check keywords without entities
        for intent_type, keywords in self.keywords.items():
            if words & keywords:
                return intent_type
        
        # Default
        return IntentType.GENERAL if entities else IntentType.UNKNOWN
    
    def _suggest_tools(self, intent_type: IntentType, entities: List[Entity]) -> List[str]:
        """Suggest tools based on intent and entities."""
        if intent_type == IntentType.FILE_READ:
            return ['read_file']
        elif intent_type == IntentType.FILE_SEARCH:
            return ['search_files', 'fuzzy_find']
        elif intent_type == IntentType.DIR_LIST:
            return ['list_directory']
        elif intent_type == IntentType.CODE_SEARCH:
            return ['search_pattern']
        elif intent_type == IntentType.GENERAL:
            # If we have entities, try to be helpful
            if any(e.type == 'file' for e in entities):
                return ['fuzzy_find', 'read_file']
            else:
                return ['search_files']
        else:
            return ['search_files']  # Safe default