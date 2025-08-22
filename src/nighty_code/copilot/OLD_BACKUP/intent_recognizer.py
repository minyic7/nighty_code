"""
Intent recognition for understanding user queries and mapping to appropriate tools.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re


class IntentType(Enum):
    """Types of user intents."""
    FILE_READ = "file_read"
    FILE_SEARCH = "file_search"
    DIRECTORY_EXPLORE = "directory_explore"
    CODE_ANALYSIS = "code_analysis"
    PATTERN_SEARCH = "pattern_search"
    FUZZY_FIND = "fuzzy_find"
    CONTEXT_SUGGEST = "context_suggest"
    GENERAL_QUERY = "general_query"
    AMBIGUOUS = "ambiguous"


@dataclass
class Intent:
    """Recognized intent from user query."""
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    suggested_tools: List[str]
    reasoning: str


class IntentRecognizer:
    """
    Recognizes user intent from natural language queries.
    Maps intents to appropriate MCP tools.
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.tool_mapping = self._initialize_tool_mapping()
    
    def _initialize_patterns(self) -> Dict[IntentType, List[Tuple[re.Pattern, float]]]:
        """Initialize regex patterns for intent recognition."""
        return {
            IntentType.FILE_READ: [
                (re.compile(r'\b(show|read|display|open|view|look at)\b.*\b(file|code|script|contents)\b', re.I), 0.9),
                (re.compile(r'\bshow me the contents of\b', re.I), 0.95),
                (re.compile(r'\bwhat\'s in\b.*\.(py|js|ts|java|go|rs|cpp|c|h|cs|rb|php)', re.I), 0.95),
                (re.compile(r'\b(get|fetch|retrieve)\b.*\bfile\b', re.I), 0.85),
            ],
            IntentType.FILE_SEARCH: [
                (re.compile(r'\b(find|search|locate|where is|look for)\b.*\b(file|files|code)\b', re.I), 0.9),
                (re.compile(r'\bfind all\b.*\.(py|js|ts|java|go|rs|cpp|c|h|cs|rb|php)', re.I), 0.95),
                (re.compile(r'\b(search for|grep|scan)\b', re.I), 0.8),
            ],
            IntentType.DIRECTORY_EXPLORE: [
                (re.compile(r'\b(list|show|explore|browse)\b.*\b(directory|folder|dir|path)\b', re.I), 0.9),
                (re.compile(r'\bwhat\'s in\b.*\b(folder|directory|dir)\b', re.I), 0.95),
                (re.compile(r'\b(ls|dir|tree)\b', re.I), 0.85),
            ],
            IntentType.CODE_ANALYSIS: [
                (re.compile(r'\b(analyze|understand|explain|review)\b.*\b(code|function|class|module)\b', re.I), 0.9),
                (re.compile(r'\bhow does\b.*\bwork\b', re.I), 0.85),
                (re.compile(r'\bwhat does\b.*\bdo\b', re.I), 0.85),
            ],
            IntentType.PATTERN_SEARCH: [
                (re.compile(r'\b(search|find|grep)\b.*\b(pattern|regex|expression)\b', re.I), 0.95),
                (re.compile(r'\bsearch for\b.*[\'\"].*[\'\"]\b.*\bin\b', re.I), 0.95),
                (re.compile(r'\bfind all\b.*\b(instances|occurrences|uses)\b', re.I), 0.9),
                (re.compile(r'\bwhere is\b.*\b(used|called|referenced)\b', re.I), 0.9),
            ],
            IntentType.FUZZY_FIND: [
                (re.compile(r'\b(similar|like|fuzzy|approximate)\b', re.I), 0.8),
                (re.compile(r'\bsomething like\b', re.I), 0.9),
                (re.compile(r'\bcan\'t remember exactly\b', re.I), 0.85),
            ],
            IntentType.CONTEXT_SUGGEST: [
                (re.compile(r'\b(suggest|recommend|what should|what next)\b', re.I), 0.85),
                (re.compile(r'\brelated\b.*\b(files|code|modules)\b', re.I), 0.9),
                (re.compile(r'\bwhat else\b', re.I), 0.8),
            ],
        }
    
    def _initialize_tool_mapping(self) -> Dict[IntentType, List[str]]:
        """Map intent types to MCP tools."""
        return {
            IntentType.FILE_READ: ["read_file"],
            IntentType.FILE_SEARCH: ["search_files", "fuzzy_find"],
            IntentType.DIRECTORY_EXPLORE: ["list_directory"],
            IntentType.CODE_ANALYSIS: ["read_file", "search_pattern"],
            IntentType.PATTERN_SEARCH: ["search_pattern"],
            IntentType.FUZZY_FIND: ["fuzzy_find"],
            IntentType.CONTEXT_SUGGEST: ["smart_suggest"],
            IntentType.GENERAL_QUERY: ["search_files", "read_file"],
            IntentType.AMBIGUOUS: ["smart_suggest", "fuzzy_find"],
        }
    
    def recognize(self, query: str) -> Intent:
        """
        Recognize intent from user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Recognized intent with confidence and suggested tools
        """
        # Extract entities from query
        entities = self._extract_entities(query)
        
        # Score each intent type
        intent_scores = []
        for intent_type, patterns in self.patterns.items():
            max_score = 0.0
            for pattern, weight in patterns:
                if pattern.search(query):
                    max_score = max(max_score, weight)
            if max_score > 0:
                intent_scores.append((intent_type, max_score))
        
        # Sort by confidence
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not intent_scores:
            # No pattern matched - treat as general query
            return Intent(
                type=IntentType.GENERAL_QUERY,
                confidence=0.5,
                entities=entities,
                suggested_tools=self.tool_mapping[IntentType.GENERAL_QUERY],
                reasoning="No specific pattern matched, treating as general query"
            )
        
        # Check if ambiguous (multiple high-confidence intents)
        if len(intent_scores) > 1 and intent_scores[1][1] >= intent_scores[0][1] * 0.9:
            return Intent(
                type=IntentType.AMBIGUOUS,
                confidence=intent_scores[0][1] * 0.7,
                entities=entities,
                suggested_tools=self._combine_tools(
                    [intent for intent, _ in intent_scores[:2]]
                ),
                reasoning=f"Query is ambiguous between {intent_scores[0][0].value} and {intent_scores[1][0].value}"
            )
        
        # Return highest confidence intent
        best_intent = intent_scores[0][0]
        return Intent(
            type=best_intent,
            confidence=intent_scores[0][1],
            entities=entities,
            suggested_tools=self.tool_mapping[best_intent],
            reasoning=self._generate_reasoning(best_intent, query)
        )
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities like file names, paths, patterns from query."""
        entities = {}
        
        # Extract file paths
        path_pattern = re.compile(r'["\']?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+)["\']?')
        paths = path_pattern.findall(query)
        if paths:
            entities['file_paths'] = paths
        
        # Extract directory paths
        dir_pattern = re.compile(r'["\']?([a-zA-Z0-9_\-./\\]+/)["\']?')
        dirs = dir_pattern.findall(query)
        if dirs:
            entities['directories'] = dirs
        
        # Extract code patterns (function/class names)
        code_pattern = re.compile(r'\b(class|function|def|method)\s+(\w+)', re.I)
        code_matches = code_pattern.findall(query)
        if code_matches:
            entities['code_elements'] = [match[1] for match in code_matches]
        
        # Extract file extensions
        ext_pattern = re.compile(r'\*?\.(py|js|ts|java|go|rs|cpp|c|h|cs|rb|php|json|yaml|yml|xml|md|txt)')
        extensions = ext_pattern.findall(query)
        if extensions:
            entities['extensions'] = extensions
        
        # Extract quoted strings (potential search terms)
        quote_pattern = re.compile(r'["\']([^"\']+)["\']')
        quoted = quote_pattern.findall(query)
        if quoted:
            entities['quoted_terms'] = quoted
        
        return entities
    
    def _combine_tools(self, intent_types: List[IntentType]) -> List[str]:
        """Combine tools from multiple intent types."""
        tools = []
        seen = set()
        for intent_type in intent_types:
            for tool in self.tool_mapping[intent_type]:
                if tool not in seen:
                    tools.append(tool)
                    seen.add(tool)
        return tools
    
    def _generate_reasoning(self, intent_type: IntentType, query: str) -> str:
        """Generate human-readable reasoning for intent recognition."""
        reasoning_templates = {
            IntentType.FILE_READ: "User wants to read or view file contents",
            IntentType.FILE_SEARCH: "User wants to search for files",
            IntentType.DIRECTORY_EXPLORE: "User wants to explore directory structure",
            IntentType.CODE_ANALYSIS: "User wants to analyze or understand code",
            IntentType.PATTERN_SEARCH: "User wants to search for specific patterns in code",
            IntentType.FUZZY_FIND: "User is looking for something but doesn't know exact name",
            IntentType.CONTEXT_SUGGEST: "User wants suggestions based on context",
            IntentType.GENERAL_QUERY: "General exploration query",
            IntentType.AMBIGUOUS: "Query has multiple possible interpretations",
        }
        return reasoning_templates.get(intent_type, "Intent recognized from query pattern")
    
    def refine_with_context(self, intent: Intent, context: Dict[str, Any]) -> Intent:
        """
        Refine intent based on conversation context.
        
        Args:
            intent: Initial recognized intent
            context: Conversation context (previous files, current topic, etc.)
            
        Returns:
            Refined intent with updated confidence and tools
        """
        # Boost confidence if context supports intent
        if 'current_file' in context and intent.type == IntentType.CONTEXT_SUGGEST:
            intent.confidence = min(1.0, intent.confidence * 1.2)
            intent.reasoning += " (boosted by current file context)"
        
        # Add context-specific entities
        if 'current_directory' in context:
            intent.entities['context_directory'] = context['current_directory']
        
        if 'recent_files' in context:
            intent.entities['recent_files'] = context['recent_files']
        
        # Adjust tools based on context
        if context.get('prefers_fuzzy_search'):
            if 'fuzzy_find' not in intent.suggested_tools:
                intent.suggested_tools.append('fuzzy_find')
        
        return intent