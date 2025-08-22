"""
Hybrid intent recognition system that combines pattern matching, entity extraction,
and LLM interpretation for robust understanding of user queries.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path

from ..llm.structured_client import StructuredLLMClient
from ..mcp.utils.fuzzy_match import FuzzyMatcher


class IntentType(Enum):
    """Types of user intents."""
    FILE_READ = "file_read"
    CODE_SEARCH = "code_search"
    EXPLAIN = "explain"
    DEBUG = "debug"
    EXPLORE = "explore"
    MODIFY = "modify"
    ANALYZE = "analyze"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    FILE_PATH = "file_path"
    FILE_NAME = "file_name"
    DIRECTORY = "directory"
    CODE_ELEMENT = "code_element"
    PATTERN = "pattern"
    LITERAL = "literal"
    EXTENSION = "extension"


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
    requires_context: bool = False
    ambiguous: bool = False


@dataclass
class ToolPlan:
    """Execution plan for tools."""
    tools: List[Dict[str, Any]]
    parallel_groups: List[List[int]] = field(default_factory=list)
    conditional_steps: Dict[int, str] = field(default_factory=dict)


class EntityExtractor:
    """Extracts entities from natural language queries."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.fuzzy_matcher = FuzzyMatcher(self.project_root) if project_root else None
        
    def extract(self, query: str) -> List[Entity]:
        """Extract all entities from query."""
        entities = []
        
        # Extract file paths (with extensions)
        file_pattern = re.compile(r'([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]{2,4})', re.I)
        for match in file_pattern.finditer(query):
            path = match.group(1)
            entities.append(Entity(
                type=EntityType.FILE_PATH if '/' in path or '\\' in path else EntityType.FILE_NAME,
                value=path,
                confidence=0.95
            ))
        
        # Extract directories
        dir_pattern = re.compile(r'\b(?:in|from|under|within)\s+([a-zA-Z0-9_\-/\\]+)/?', re.I)
        for match in dir_pattern.finditer(query):
            entities.append(Entity(
                type=EntityType.DIRECTORY,
                value=match.group(1),
                confidence=0.85
            ))
        
        # Extract quoted literals (high confidence)
        quote_pattern = re.compile(r'["\']([^"\']+)["\']')
        for match in quote_pattern.finditer(query):
            literal = match.group(1)
            # Determine if it's a pattern or literal
            entity_type = EntityType.PATTERN if any(c in literal for c in ['*', '?', '[', ']']) else EntityType.LITERAL
            entities.append(Entity(
                type=entity_type,
                value=literal,
                confidence=0.98
            ))
        
        # Extract code elements
        code_pattern = re.compile(r'\b(class|function|method|def|interface|struct|enum)\s+(\w+)', re.I)
        for match in code_pattern.finditer(query):
            entities.append(Entity(
                type=EntityType.CODE_ELEMENT,
                value=match.group(2),
                metadata={'element_type': match.group(1).lower()},
                confidence=0.9
            ))
        
        # Extract file extensions
        ext_pattern = re.compile(r'\*?\.(py|js|ts|java|go|rs|cpp|c|h|cs|rb|php|json|yaml|yml|xml|md|txt)\b', re.I)
        for match in ext_pattern.finditer(query):
            entities.append(Entity(
                type=EntityType.EXTENSION,
                value=match.group(1).lower(),
                confidence=0.95
            ))
        
        # Validate and enhance entities with fuzzy matching
        if self.fuzzy_matcher:
            entities = self._enhance_with_fuzzy_matching(entities)
        
        return entities
    
    def _enhance_with_fuzzy_matching(self, entities: List[Entity]) -> List[Entity]:
        """Enhance entities with fuzzy matching to validate and find similar files."""
        enhanced = []
        
        for entity in entities:
            if entity.type in [EntityType.FILE_PATH, EntityType.FILE_NAME]:
                # Check if file exists or find similar
                file_path = self.project_root / entity.value
                if file_path.exists():
                    entity.confidence = 1.0
                    entity.metadata['exists'] = True
                else:
                    # Try fuzzy matching
                    matches = self.fuzzy_matcher.find_similar_files(entity.value, limit=3)
                    if matches:
                        entity.metadata['suggestions'] = [m['path'] for m in matches]
                        entity.metadata['exists'] = False
                        entity.confidence *= 0.7  # Lower confidence for non-existent files
            
            enhanced.append(entity)
        
        return enhanced


class KeywordAnalyzer:
    """Analyzes keywords to understand query intent."""
    
    # Action keywords mapped to intent types
    ACTION_KEYWORDS = {
        IntentType.FILE_READ: {'show', 'read', 'display', 'view', 'open', 'see', 'look', 'contents'},
        IntentType.CODE_SEARCH: {'find', 'search', 'locate', 'where', 'grep', 'scan', 'look for'},
        IntentType.EXPLAIN: {'explain', 'understand', 'how', 'why', 'what', 'describe', 'tell'},
        IntentType.DEBUG: {'debug', 'fix', 'error', 'bug', 'issue', 'problem', 'wrong', 'broken'},
        IntentType.EXPLORE: {'explore', 'browse', 'list', 'navigate', 'structure', 'tree', 'overview'},
        IntentType.MODIFY: {'change', 'modify', 'update', 'edit', 'refactor', 'rename', 'move'},
        IntentType.ANALYZE: {'analyze', 'review', 'check', 'validate', 'test', 'profile', 'measure'}
    }
    
    def analyze(self, query: str) -> Tuple[List[str], Dict[IntentType, float]]:
        """
        Analyze keywords in query.
        
        Returns:
            Tuple of (keywords found, intent scores)
        """
        query_lower = query.lower()
        words = set(query_lower.split())
        
        keywords_found = []
        intent_scores = {}
        
        for intent_type, keywords in self.ACTION_KEYWORDS.items():
            matching_keywords = words.intersection(keywords)
            if matching_keywords:
                keywords_found.extend(matching_keywords)
                # Score based on number of matching keywords (not total keywords in category)
                score = len(matching_keywords) * 0.3  # Base score per keyword
                # Boost score if keyword appears early in query
                for keyword in matching_keywords:
                    if query_lower.startswith(keyword):
                        score += 0.2
                intent_scores[intent_type] = min(score, 1.0)
        
        return list(set(keywords_found)), intent_scores


class HybridIntentRecognizer:
    """
    Hybrid intent recognition combining multiple strategies.
    Uses patterns for obvious cases, entity extraction for context,
    and LLM for complex understanding.
    """
    
    def __init__(self, llm_client: Optional[StructuredLLMClient] = None, project_root: Optional[Path] = None):
        self.llm_client = llm_client
        self.entity_extractor = EntityExtractor(project_root)
        self.keyword_analyzer = KeywordAnalyzer()
        self.project_root = project_root or Path.cwd()
        
        # Only the most unambiguous patterns
        self.obvious_patterns = [
            (re.compile(r'^read\s+(\S+\.py)$', re.I), IntentType.FILE_READ),
            (re.compile(r'^list\s+files\s+in\s+(\S+)/?$', re.I), IntentType.EXPLORE),
            (re.compile(r'^search\s+for\s+["\']([^"\']+)["\']$', re.I), IntentType.CODE_SEARCH),
        ]
    
    def recognize(self, query: str, context: Optional[Dict[str, Any]] = None) -> ProcessedIntent:
        """
        Recognize intent from query using hybrid approach.
        
        Args:
            query: User's natural language query
            context: Optional context from conversation
            
        Returns:
            Processed intent with confidence and extracted information
        """
        # 1. Extract entities first - they're useful regardless of intent
        entities = self.entity_extractor.extract(query)
        
        # 2. Analyze keywords
        keywords, intent_scores = self.keyword_analyzer.analyze(query)
        
        # 3. Check obvious patterns
        if obvious_intent := self._check_obvious_patterns(query):
            return ProcessedIntent(
                type=obvious_intent,
                confidence=0.95,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(obvious_intent, entities),
                reasoning="Matched unambiguous pattern",
                ambiguous=False
            )
        
        # 4. Try to infer from entities and keywords
        if inferred := self._infer_from_signals(query, entities, keywords, intent_scores):
            if inferred.confidence >= 0.7:
                return inferred
        
        # 5. Use LLM for complex understanding
        if self.llm_client:
            return self._llm_interpret(query, entities, keywords, intent_scores, context)
        
        # 6. Fallback to best guess from keyword analysis
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return ProcessedIntent(
                type=best_intent[0],
                confidence=best_intent[1] * 0.6,  # Lower confidence for keyword-only
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(best_intent[0], entities),
                reasoning=f"Best guess from keywords: {keywords}",
                ambiguous=True
            )
        
        # 7. Complete unknown
        return ProcessedIntent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            entities=entities,
            keywords=keywords,
            suggested_tools=['smart_suggest', 'fuzzy_find'],
            reasoning="Could not determine intent",
            ambiguous=True
        )
    
    def _check_obvious_patterns(self, query: str) -> Optional[IntentType]:
        """Check only the most obvious, unambiguous patterns."""
        for pattern, intent_type in self.obvious_patterns:
            if pattern.match(query.strip()):
                return intent_type
        return None
    
    def _infer_from_signals(
        self, 
        query: str, 
        entities: List[Entity], 
        keywords: List[str],
        intent_scores: Dict[IntentType, float]
    ) -> Optional[ProcessedIntent]:
        """Infer intent from extracted signals."""
        
        # Strong signals combinations
        has_file = any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities)
        has_pattern = any(e.type == EntityType.PATTERN for e in entities)
        has_literal = any(e.type == EntityType.LITERAL for e in entities)
        has_directory = any(e.type == EntityType.DIRECTORY for e in entities)
        
        # Clear combinations
        if has_file and 'read' in keywords:
            return ProcessedIntent(
                type=IntentType.FILE_READ,
                confidence=0.85,
                entities=entities,
                keywords=keywords,
                suggested_tools=['read_file'],
                reasoning="Has file reference + read keyword"
            )
        
        if (has_pattern or has_literal) and any(k in keywords for k in ['search', 'find', 'grep']):
            return ProcessedIntent(
                type=IntentType.CODE_SEARCH,
                confidence=0.8,
                entities=entities,
                keywords=keywords,
                suggested_tools=['search_pattern', 'search_files'],
                reasoning="Has search pattern/literal + search keyword"
            )
        
        if has_directory and any(k in keywords for k in ['list', 'explore', 'browse']):
            return ProcessedIntent(
                type=IntentType.EXPLORE,
                confidence=0.8,
                entities=entities,
                keywords=keywords,
                suggested_tools=['list_directory'],
                reasoning="Has directory + explore keyword"
            )
        
        # Use intent scores if one is significantly higher
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_intents[0][1] >= 0.3:  # Lower threshold for keyword-based
                # Adjust confidence based on entity support
                confidence = sorted_intents[0][1]
                if entities:
                    confidence = min(confidence + 0.2, 1.0)  # Boost if we have entities
                
                return ProcessedIntent(
                    type=sorted_intents[0][0],
                    confidence=confidence,
                    entities=entities,
                    keywords=keywords,
                    suggested_tools=self._suggest_tools_for_intent(sorted_intents[0][0], entities),
                    reasoning=f"Keyword and entity analysis: {sorted_intents[0][0].value}",
                    ambiguous=len(sorted_intents) > 1 and sorted_intents[1][1] > sorted_intents[0][1] * 0.7
                )
        
        return None
    
    def _llm_interpret(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
        intent_scores: Dict[IntentType, float],
        context: Optional[Dict[str, Any]]
    ) -> ProcessedIntent:
        """Use LLM for nuanced intent understanding."""
        
        # Build structured prompt
        prompt = f"""Analyze this user query and determine their intent.

Query: "{query}"

Extracted Information:
- Entities: {self._format_entities(entities)}
- Keywords: {keywords}
- Initial scores: {self._format_scores(intent_scores)}
- Context: {self._format_context(context)}

Possible intents:
- FILE_READ: User wants to see file contents
- CODE_SEARCH: User wants to find code/files
- EXPLAIN: User wants understanding/explanation  
- DEBUG: User wants to fix something
- EXPLORE: User wants to browse/discover
- MODIFY: User wants to change something
- ANALYZE: User wants to review/check code

Based on the query and extracted information, what is the user's most likely intent?
Consider:
1. The specific words used
2. The entities mentioned
3. The overall context
4. Any ambiguity that needs clarification

Return a JSON object with:
{{
  "intent": "INTENT_TYPE",
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "suggested_tools": ["tool1", "tool2"],
  "needs_clarification": true/false,
  "clarification_question": "optional question to ask user"
}}"""

        try:
            response = self.llm_client.complete_structured(
                prompt=prompt,
                response_format={
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                        "suggested_tools": {"type": "array", "items": {"type": "string"}},
                        "needs_clarification": {"type": "boolean"},
                        "clarification_question": {"type": "string"}
                    },
                    "required": ["intent", "confidence", "reasoning", "suggested_tools"]
                }
            )
            
            intent_type = IntentType[response['intent']]
            
            return ProcessedIntent(
                type=intent_type,
                confidence=response['confidence'],
                entities=entities,
                keywords=keywords,
                suggested_tools=response['suggested_tools'],
                reasoning=response['reasoning'],
                ambiguous=response.get('needs_clarification', False)
            )
            
        except Exception as e:
            # Fallback if LLM fails
            return ProcessedIntent(
                type=IntentType.UNKNOWN,
                confidence=0.3,
                entities=entities,
                keywords=keywords,
                suggested_tools=['smart_suggest'],
                reasoning=f"LLM interpretation failed: {e}",
                ambiguous=True
            )
    
    def _suggest_tools_for_intent(self, intent_type: IntentType, entities: List[Entity]) -> List[str]:
        """Suggest appropriate tools based on intent and entities."""
        
        tool_mapping = {
            IntentType.FILE_READ: ['read_file', 'fuzzy_find'],
            IntentType.CODE_SEARCH: ['search_pattern', 'search_files', 'fuzzy_find'],
            IntentType.EXPLAIN: ['read_file', 'search_pattern'],
            IntentType.DEBUG: ['search_pattern', 'read_file'],
            IntentType.EXPLORE: ['list_directory', 'search_files'],
            IntentType.MODIFY: ['read_file', 'search_files'],
            IntentType.ANALYZE: ['read_file', 'search_pattern', 'list_directory'],
            IntentType.UNKNOWN: ['smart_suggest', 'fuzzy_find']
        }
        
        tools = tool_mapping.get(intent_type, ['smart_suggest'])
        
        # Adjust based on entities
        if not any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities):
            # No specific file mentioned, might need to find it first
            if 'fuzzy_find' not in tools:
                tools.insert(0, 'fuzzy_find')
        
        return tools
    
    def _format_entities(self, entities: List[Entity]) -> str:
        """Format entities for prompt."""
        if not entities:
            return "None"
        return ", ".join([f"{e.type.value}:{e.value}" for e in entities])
    
    def _format_scores(self, scores: Dict[IntentType, float]) -> str:
        """Format intent scores for prompt."""
        if not scores:
            return "None"
        return ", ".join([f"{k.value}:{v:.2f}" for k, v in scores.items()])
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for prompt."""
        if not context:
            return "No context"
        relevant = []
        if 'current_file' in context:
            relevant.append(f"Current file: {context['current_file']}")
        if 'recent_files' in context:
            relevant.append(f"Recent files: {context['recent_files'][:3]}")
        return ", ".join(relevant) if relevant else "No relevant context"


class IntentToToolMapper:
    """Maps processed intents to executable tool chains."""
    
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        self.available_tools = self._get_available_tools()
    
    def _get_available_tools(self) -> Set[str]:
        """Get list of available MCP tools."""
        # This would call mcp_server.list_tools() in real implementation
        return {
            'read_file', 'list_directory', 'search_files', 'search_pattern',
            'fuzzy_find', 'smart_suggest'
        }
    
    def map(self, intent: ProcessedIntent, context: Optional[Dict[str, Any]] = None) -> ToolPlan:
        """
        Map processed intent to executable tool plan.
        
        Args:
            intent: Processed intent with entities
            context: Optional execution context
            
        Returns:
            Tool execution plan
        """
        tools = []
        
        if intent.type == IntentType.FILE_READ:
            # Check if we have a specific file
            file_entities = [e for e in intent.entities if e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME]]
            
            if file_entities and file_entities[0].metadata.get('exists'):
                # Direct read
                tools.append({
                    'name': 'read_file',
                    'params': {'file_path': file_entities[0].value}
                })
            elif file_entities:
                # Need to find file first
                tools.append({
                    'name': 'fuzzy_find',
                    'params': {'query': file_entities[0].value}
                })
                tools.append({
                    'name': 'read_file',
                    'params': {'file_path': '{previous_result[0]}'},
                    'depends_on': 0
                })
            else:
                # No file specified, need more context
                tools.append({
                    'name': 'smart_suggest',
                    'params': {'context': context}
                })
        
        elif intent.type == IntentType.CODE_SEARCH:
            pattern_entities = [e for e in intent.entities if e.type in [EntityType.PATTERN, EntityType.LITERAL]]
            
            if pattern_entities:
                tools.append({
                    'name': 'search_pattern',
                    'params': {'pattern': pattern_entities[0].value}
                })
            else:
                # Broader search
                tools.append({
                    'name': 'search_files',
                    'params': {'query': ' '.join(intent.keywords)}
                })
        
        elif intent.type == IntentType.EXPLORE:
            dir_entities = [e for e in intent.entities if e.type == EntityType.DIRECTORY]
            
            tools.append({
                'name': 'list_directory',
                'params': {'directory_path': dir_entities[0].value if dir_entities else '.'}
            })
        
        else:
            # For complex intents, use suggested tools
            for tool_name in intent.suggested_tools:
                if tool_name in self.available_tools:
                    tools.append({
                        'name': tool_name,
                        'params': self._build_params_for_tool(tool_name, intent, context)
                    })
        
        return ToolPlan(tools=tools)
    
    def _build_params_for_tool(
        self, 
        tool_name: str, 
        intent: ProcessedIntent, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build parameters for a specific tool based on intent."""
        params = {}
        
        if tool_name == 'read_file':
            file_entities = [e for e in intent.entities if e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME]]
            if file_entities:
                params['file_path'] = file_entities[0].value
        
        elif tool_name == 'search_pattern':
            pattern_entities = [e for e in intent.entities if e.type in [EntityType.PATTERN, EntityType.LITERAL]]
            if pattern_entities:
                params['pattern'] = pattern_entities[0].value
        
        elif tool_name == 'list_directory':
            dir_entities = [e for e in intent.entities if e.type == EntityType.DIRECTORY]
            params['directory_path'] = dir_entities[0].value if dir_entities else '.'
        
        elif tool_name == 'fuzzy_find':
            # Use any file-like entity or literal
            search_entities = [e for e in intent.entities 
                             if e.type in [EntityType.FILE_NAME, EntityType.LITERAL, EntityType.CODE_ELEMENT]]
            if search_entities:
                params['query'] = search_entities[0].value
        
        elif tool_name == 'smart_suggest':
            params['context'] = context or {}
        
        return params