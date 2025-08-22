"""
Production-ready intent recognition system with comprehensive error handling,
monitoring, caching, and graceful degradation.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Deque
from datetime import datetime, timedelta
import threading

from ..llm.structured_client import StructuredLLMClient
from ..mcp.utils.fuzzy_match import FuzzyMatcher


logger = logging.getLogger(__name__)


# Configuration
@dataclass
class RecognizerConfig:
    """Configuration for production intent recognizer."""
    # Limits
    max_query_length: int = 1000
    max_entities: int = 20
    max_context_size: int = 5000
    
    # Timeouts (seconds)
    entity_extraction_timeout: float = 2.0
    keyword_analysis_timeout: float = 1.0
    llm_timeout: float = 10.0
    total_timeout: float = 15.0
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_expected_exception: type = Exception
    
    # Rate limiting
    max_requests_per_minute: int = 60
    
    # Confidence thresholds (configurable!)
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.5
    low_confidence_threshold: float = 0.3
    entity_confidence_boost: float = 0.2
    keyword_base_score: float = 0.3
    keyword_position_boost: float = 0.2
    
    # Feature flags
    enable_llm_fallback: bool = True
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_audit_logging: bool = True
    
    # Security
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        'py', 'js', 'ts', 'java', 'go', 'rs', 'cpp', 'c', 'h', 'cs', 'rb', 
        'php', 'json', 'yaml', 'yml', 'xml', 'md', 'txt', 'toml', 'ini'
    })
    max_path_depth: int = 10
    sanitize_inputs: bool = True


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for external dependencies."""
    
    def __init__(self, config: RecognizerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if we should try recovery
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.circuit_breaker_recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.config.circuit_breaker_failure_threshold:
                self.state = CircuitBreakerState.OPEN


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = max_per_minute
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a token."""
        with self.lock:
            self._refill()
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        # Only refill if enough time has passed (at least 1 second)
        if elapsed >= 1.0:
            tokens_to_add = int(elapsed * (self.max_per_minute / 60.0))
            if tokens_to_add > 0:
                self.tokens = min(self.max_per_minute, self.tokens + tokens_to_add)
                self.last_refill = now


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: Any
    expires_at: float
    hit_count: int = 0


class LRUCache:
    """LRU cache with TTL support."""
    
    def __init__(self, maxsize: int, ttl: int):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: Deque[str] = deque(maxlen=maxsize)
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if time.time() > entry.expires_at:
                # Expired
                del self.cache[key]
                return None
            
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            entry.hit_count += 1
            
            return entry.value
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.maxsize and key not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + self.ttl
            )
            
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)


class MetricsCollector:
    """Collect and report metrics."""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'llm_failures': 0,
            'timeouts': 0,
            'average_latency': 0.0,
            'intent_distribution': {},
            'error_types': {}
        }
        self.latencies: Deque[float] = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def record_request(self, intent_type: str, latency: float, success: bool):
        """Record a request."""
        with self.lock:
            self.metrics['total_requests'] += 1
            if success:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            self.latencies.append(latency)
            self.metrics['average_latency'] = sum(self.latencies) / len(self.latencies)
            
            if intent_type not in self.metrics['intent_distribution']:
                self.metrics['intent_distribution'][intent_type] = 0
            self.metrics['intent_distribution'][intent_type] += 1
    
    def record_cache_hit(self):
        """Record cache hit."""
        with self.lock:
            self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        with self.lock:
            self.metrics['cache_misses'] += 1
    
    def record_error(self, error_type: str):
        """Record error."""
        with self.lock:
            if error_type not in self.metrics['error_types']:
                self.metrics['error_types'][error_type] = 0
            self.metrics['error_types'][error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            return self.metrics.copy()


class AuditLogger:
    """Audit logger for tracking all operations."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger('audit')
    
    def log_request(self, request_id: str, query: str, context: Optional[Dict]):
        """Log incoming request."""
        if not self.enabled:
            return
        
        self.logger.info(json.dumps({
            'event': 'request',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'query_length': len(query),
            'has_context': context is not None
        }))
    
    def log_intent(self, request_id: str, intent_type: str, confidence: float, method: str):
        """Log recognized intent."""
        if not self.enabled:
            return
        
        self.logger.info(json.dumps({
            'event': 'intent_recognized',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'intent_type': intent_type,
            'confidence': confidence,
            'recognition_method': method
        }))
    
    def log_tool_execution(self, request_id: str, tool_name: str, params: Dict, success: bool):
        """Log tool execution."""
        if not self.enabled:
            return
        
        self.logger.info(json.dumps({
            'event': 'tool_execution',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'tool_name': tool_name,
            'param_keys': list(params.keys()),
            'success': success
        }))


class InputValidator:
    """Validate and sanitize user inputs."""
    
    def __init__(self, config: RecognizerConfig):
        self.config = config
        # Pre-compile dangerous patterns
        self.dangerous_patterns = [
            re.compile(r'\.\.[\\/]'),  # Path traversal
            re.compile(r'[;&|`$]'),     # Command injection
            re.compile(r'<script', re.I),  # XSS
            re.compile(r'DROP\s+TABLE', re.I),  # SQL injection
            re.compile(r"'\s*OR\s*'", re.I),  # SQL injection pattern
            re.compile(r'UNION\s+SELECT', re.I),  # SQL injection
        ]
    
    def validate_query(self, query: str) -> str:
        """Validate and sanitize query."""
        if not query:
            raise ValueError("Query cannot be empty")
        
        if len(query) > self.config.max_query_length:
            raise ValueError(f"Query too long: {len(query)} > {self.config.max_query_length}")
        
        if self.config.sanitize_inputs:
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if pattern.search(query):
                    raise ValueError("Query contains potentially dangerous content")
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query
    
    def validate_path(self, path: str) -> str:
        """Validate file path."""
        # Check for null bytes first
        if '\x00' in path:
            raise ValueError("Path contains null bytes")
        
        # Remove any null bytes as safety
        path = path.replace('\x00', '')
        
        # Check depth
        parts = Path(path).parts
        if len(parts) > self.config.max_path_depth:
            raise ValueError(f"Path too deep: {len(parts)} > {self.config.max_path_depth}")
        
        # Check for path traversal
        if '..' in parts:
            raise ValueError("Path traversal detected")
        
        # Check extension
        if '.' in path:
            ext = path.split('.')[-1].lower()
            if ext not in self.config.allowed_file_extensions:
                raise ValueError(f"File extension not allowed: {ext}")
        
        return path


# Pre-compiled regex patterns for performance
class CompiledPatterns:
    """Pre-compiled regex patterns."""
    
    def __init__(self):
        # Entity patterns
        self.file_path = re.compile(r'([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]{2,4})', re.I)
        self.directory = re.compile(r'\b(?:in|from|under|within)\s+([a-zA-Z0-9_\-/\\]+)/?', re.I)
        self.quoted = re.compile(r'["\']([^"\']+)["\']')
        self.code_element = re.compile(r'\b(class|function|method|def|interface|struct|enum)\s+(\w+)', re.I)
        self.extension = re.compile(r'\*?\.(py|js|ts|java|go|rs|cpp|c|h|cs|rb|php|json|yaml|yml|xml|md|txt)\b', re.I)
        
        # Obvious intent patterns
        self.obvious_read = re.compile(r'^read\s+(\S+\.\w+)$', re.I)
        self.obvious_list = re.compile(r'^list\s+files\s+in\s+(\S+)/?$', re.I)
        self.obvious_search = re.compile(r'^search\s+for\s+["\']([^"\']+)["\']$', re.I)


# Import intent types from existing module
from .hybrid_intent import IntentType, EntityType, Entity, ProcessedIntent


class RobustIntentRecognizer:
    """
    Production-ready intent recognizer with comprehensive error handling,
    monitoring, caching, and graceful degradation.
    """
    
    def __init__(
        self,
        config: Optional[RecognizerConfig] = None,
        llm_client: Optional[StructuredLLMClient] = None,
        project_root: Optional[Path] = None
    ):
        self.config = config or RecognizerConfig()
        self.llm_client = llm_client
        self.project_root = project_root or Path.cwd()
        
        # Pre-compile patterns for efficiency
        self.patterns = CompiledPatterns()
        
        # Initialize components
        self.cache = LRUCache(self.config.cache_size, self.config.cache_ttl)
        self.metrics = MetricsCollector()
        self.audit = AuditLogger(self.config.enable_audit_logging)
        self.validator = InputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
        self.circuit_breaker = CircuitBreaker(self.config)
        
        # Fuzzy matcher if project root exists
        try:
            if self.project_root.exists():
                self.fuzzy_matcher = FuzzyMatcher(self.project_root)
            else:
                self.fuzzy_matcher = None
                logger.warning(f"Project root does not exist: {self.project_root}")
        except Exception as e:
            self.fuzzy_matcher = None
            logger.error(f"Failed to initialize fuzzy matcher: {e}")
        
        logger.info("RobustIntentRecognizer initialized")
    
    async def recognize(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> ProcessedIntent:
        """
        Recognize intent from query with full production safeguards.
        
        Args:
            query: User's natural language query
            context: Optional context from conversation
            request_id: Optional request ID for tracing
            
        Returns:
            Processed intent with confidence and extracted information
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Audit logging
        self.audit.log_request(request_id, query, context)
        
        try:
            # Rate limiting
            if not self.rate_limiter.acquire():
                self.metrics.record_error('rate_limit')
                raise Exception("Rate limit exceeded")
            
            # Input validation
            try:
                query = self.validator.validate_query(query)
            except ValueError as e:
                self.metrics.record_error('validation')
                logger.warning(f"Query validation failed: {e}")
                return self._create_error_intent("Invalid query", str(e))
            
            # Check cache
            cache_key = self._get_cache_key(query, context)
            if self.config.enable_caching:
                if cached := self.cache.get(cache_key):
                    self.metrics.record_cache_hit()
                    self.metrics.record_request(cached.type.value, time.time() - start_time, True)
                    return cached
                else:
                    self.metrics.record_cache_miss()
            
            # Recognize with timeout protection
            try:
                async with asyncio.timeout(self.config.total_timeout):
                    intent = await self._recognize_internal(query, context, request_id)
            except asyncio.TimeoutError:
                self.metrics.record_error('timeout')
                logger.warning(f"Recognition timeout for query: {query[:50]}...")
                intent = self._get_fallback_intent(query, "Recognition timeout")
            
            # Cache successful result
            if self.config.enable_caching and intent.confidence >= self.config.low_confidence_threshold:
                self.cache.put(cache_key, intent)
            
            # Record metrics
            latency = time.time() - start_time
            self.metrics.record_request(intent.type.value, latency, True)
            
            # Audit logging
            self.audit.log_intent(request_id, intent.type.value, intent.confidence, intent.reasoning)
            
            return intent
            
        except Exception as e:
            # Record failure
            self.metrics.record_error(type(e).__name__)
            self.metrics.record_request('error', time.time() - start_time, False)
            logger.error(f"Recognition failed for request {request_id}: {e}")
            
            # Return safe fallback
            return self._get_fallback_intent(query, str(e))
    
    async def _recognize_internal(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> ProcessedIntent:
        """Internal recognition logic with error handling."""
        
        # 1. Extract entities with timeout
        entities = await self._extract_entities_safe(query)
        
        # 2. Analyze keywords with timeout
        keywords, intent_scores = await self._analyze_keywords_safe(query)
        
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
        
        # 4. Try inference from signals
        if inferred := self._infer_from_signals(query, entities, keywords, intent_scores):
            if inferred.confidence >= self.config.medium_confidence_threshold:
                return inferred
        
        # 5. Try LLM if available and enabled
        if self.config.enable_llm_fallback and self.llm_client:
            try:
                llm_intent = await self._llm_interpret_safe(
                    query, entities, keywords, intent_scores, context, request_id
                )
                if llm_intent and llm_intent.confidence >= self.config.low_confidence_threshold:
                    return llm_intent
            except Exception as e:
                logger.warning(f"LLM interpretation failed: {e}")
                # Continue with fallback
        
        # 6. Best effort from keywords
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return ProcessedIntent(
                type=best_intent[0],
                confidence=best_intent[1] * 0.5,  # Lower confidence
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(best_intent[0], entities),
                reasoning="Best guess from keywords",
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
    
    async def _extract_entities_safe(self, query: str) -> List[Entity]:
        """Extract entities with timeout and error handling."""
        try:
            async with asyncio.timeout(self.config.entity_extraction_timeout):
                return await asyncio.to_thread(self._extract_entities, query)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities using pre-compiled patterns."""
        entities = []
        
        try:
            # File paths
            for match in self.patterns.file_path.finditer(query):
                path = match.group(1)
                # Validate path
                try:
                    path = self.validator.validate_path(path)
                    entity_type = EntityType.FILE_PATH if '/' in path or '\\' in path else EntityType.FILE_NAME
                    entities.append(Entity(
                        type=entity_type,
                        value=path,
                        confidence=0.95
                    ))
                except ValueError:
                    continue  # Skip invalid paths
            
            # Limit entities to prevent DoS
            if len(entities) > self.config.max_entities:
                entities = entities[:self.config.max_entities]
            
            # Directories
            for match in self.patterns.directory.finditer(query):
                if len(entities) < self.config.max_entities:
                    entities.append(Entity(
                        type=EntityType.DIRECTORY,
                        value=match.group(1),
                        confidence=0.85
                    ))
            
            # Quoted literals
            for match in self.patterns.quoted.finditer(query):
                if len(entities) < self.config.max_entities:
                    literal = match.group(1)
                    entity_type = EntityType.PATTERN if any(c in literal for c in ['*', '?', '[', ']']) else EntityType.LITERAL
                    entities.append(Entity(
                        type=entity_type,
                        value=literal,
                        confidence=0.98
                    ))
            
            # Code elements
            for match in self.patterns.code_element.finditer(query):
                if len(entities) < self.config.max_entities:
                    entities.append(Entity(
                        type=EntityType.CODE_ELEMENT,
                        value=match.group(2),
                        metadata={'element_type': match.group(1).lower()},
                        confidence=0.9
                    ))
            
            # Extensions
            for match in self.patterns.extension.finditer(query):
                if len(entities) < self.config.max_entities:
                    entities.append(Entity(
                        type=EntityType.EXTENSION,
                        value=match.group(1).lower(),
                        confidence=0.95
                    ))
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
        
        return entities
    
    async def _analyze_keywords_safe(self, query: str) -> Tuple[List[str], Dict[IntentType, float]]:
        """Analyze keywords with timeout and error handling."""
        try:
            async with asyncio.timeout(self.config.keyword_analysis_timeout):
                return await asyncio.to_thread(self._analyze_keywords, query)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Keyword analysis failed: {e}")
            return [], {}
    
    def _analyze_keywords(self, query: str) -> Tuple[List[str], Dict[IntentType, float]]:
        """Analyze keywords for intent signals."""
        # This would use the keyword analyzer from hybrid_intent
        # Simplified for brevity
        keywords = []
        intent_scores = {}
        
        query_lower = query.lower()
        words = set(query_lower.split())
        
        # Check for key action words
        action_keywords = {
            IntentType.FILE_READ: {'show', 'read', 'display', 'view', 'open'},
            IntentType.CODE_SEARCH: {'find', 'search', 'locate', 'grep'},
            IntentType.EXPLAIN: {'explain', 'how', 'why', 'what'},
            IntentType.EXPLORE: {'list', 'browse', 'explore'},
        }
        
        for intent_type, action_words in action_keywords.items():
            matching = words.intersection(action_words)
            if matching:
                keywords.extend(matching)
                score = len(matching) * self.config.keyword_base_score
                # Position boost
                for word in matching:
                    if query_lower.startswith(word):
                        score += self.config.keyword_position_boost
                intent_scores[intent_type] = min(score, 1.0)
        
        return list(set(keywords)), intent_scores
    
    def _check_obvious_patterns(self, query: str) -> Optional[IntentType]:
        """Check obvious patterns using pre-compiled regex."""
        query = query.strip()
        
        if self.patterns.obvious_read.match(query):
            return IntentType.FILE_READ
        if self.patterns.obvious_list.match(query):
            return IntentType.EXPLORE
        if self.patterns.obvious_search.match(query):
            return IntentType.CODE_SEARCH
        
        return None
    
    def _infer_from_signals(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
        intent_scores: Dict[IntentType, float]
    ) -> Optional[ProcessedIntent]:
        """Infer intent from extracted signals."""
        
        # Check entity-keyword combinations
        has_file = any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities)
        has_pattern = any(e.type == EntityType.PATTERN for e in entities)
        has_directory = any(e.type == EntityType.DIRECTORY for e in entities)
        
        # Strong combinations
        if has_file and 'read' in keywords:
            return ProcessedIntent(
                type=IntentType.FILE_READ,
                confidence=0.85,
                entities=entities,
                keywords=keywords,
                suggested_tools=['read_file'],
                reasoning="File reference with read keyword"
            )
        
        if has_pattern and any(k in keywords for k in ['search', 'find']):
            return ProcessedIntent(
                type=IntentType.CODE_SEARCH,
                confidence=0.8,
                entities=entities,
                keywords=keywords,
                suggested_tools=['search_pattern'],
                reasoning="Pattern with search keyword"
            )
        
        # Use intent scores
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_intents[0][1] >= self.config.medium_confidence_threshold:
                confidence = sorted_intents[0][1]
                if entities:
                    confidence = min(confidence + self.config.entity_confidence_boost, 1.0)
                
                return ProcessedIntent(
                    type=sorted_intents[0][0],
                    confidence=confidence,
                    entities=entities,
                    keywords=keywords,
                    suggested_tools=self._suggest_tools_for_intent(sorted_intents[0][0], entities),
                    reasoning="Keyword and entity analysis",
                    ambiguous=len(sorted_intents) > 1 and sorted_intents[1][1] > sorted_intents[0][1] * 0.7
                )
        
        return None
    
    async def _llm_interpret_safe(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
        intent_scores: Dict[IntentType, float],
        context: Optional[Dict[str, Any]],
        request_id: str
    ) -> Optional[ProcessedIntent]:
        """LLM interpretation with circuit breaker and error handling."""
        try:
            # Use circuit breaker
            async def llm_call():
                async with asyncio.timeout(self.config.llm_timeout):
                    return await self._llm_interpret(
                        query, entities, keywords, intent_scores, context
                    )
            
            result = await asyncio.to_thread(
                self.circuit_breaker.call,
                asyncio.run,
                llm_call()
            )
            
            self.metrics.metrics['llm_calls'] += 1
            return result
            
        except Exception as e:
            self.metrics.metrics['llm_failures'] += 1
            logger.error(f"LLM interpretation failed for {request_id}: {e}")
            return None
    
    async def _llm_interpret(
        self,
        query: str,
        entities: List[Entity],
        keywords: List[str],
        intent_scores: Dict[IntentType, float],
        context: Optional[Dict[str, Any]]
    ) -> ProcessedIntent:
        """LLM interpretation for ambiguous queries."""
        if not self.llm_client:
            # No LLM available, return best guess
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else IntentType.UNKNOWN
            return ProcessedIntent(
                type=best_intent,
                confidence=0.3,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(best_intent, entities),
                reasoning="No LLM available, using pattern matching",
                ambiguous=True
            )
        
        # Build prompt for LLM
        prompt = f"""Interpret this user query for a code assistant tool.

Query: "{query}"

Context:
- Keywords found: {', '.join(keywords) if keywords else 'none'}
- Entities detected: {', '.join(f"{e.type}:{e.value}" for e in entities) if entities else 'none'}
- Possible intents: {', '.join(f"{k.value}({v:.1f})" for k, v in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:3]) if intent_scores else 'none'}

Determine the most likely intent from these options:
- FILE_READ: User wants to read/view a specific file
- CODE_SEARCH: User wants to search for code patterns
- EXPLAIN: User wants explanation about code
- EXPLORE: User wants to browse/list files or directories  
- UNKNOWN: Query doesn't match any clear intent

Respond with:
1. The intent type (one of the above)
2. A brief reasoning (one sentence)
3. Whether the query is ambiguous (yes/no)

Format: INTENT|reasoning|ambiguous"""

        try:
            # Create a Pydantic model for the response
            from pydantic import BaseModel
            
            class IntentResponse(BaseModel):
                intent: str
                reasoning: str
                ambiguous: bool
            
            # Use structured LLM client to get response
            # Note: generate is synchronous, so we need to run it in executor
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.llm_client.generate,
                prompt,
                IntentResponse
            )
            
            # Parse response (it's a Pydantic model, not a dict)
            intent_str = response.intent.upper()
            intent_type = IntentType[intent_str] if intent_str in IntentType.__members__ else IntentType.UNKNOWN
            reasoning = response.reasoning
            ambiguous = response.ambiguous
            
            # Calculate confidence based on LLM response
            confidence = 0.8 if intent_type != IntentType.UNKNOWN else 0.4
            if ambiguous:
                confidence *= 0.7
            
            return ProcessedIntent(
                type=intent_type,
                confidence=confidence,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(intent_type, entities),
                reasoning=reasoning,
                ambiguous=ambiguous
            )
            
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            # Fallback to best pattern match
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else IntentType.UNKNOWN
            return ProcessedIntent(
                type=best_intent,
                confidence=0.3,
                entities=entities,
                keywords=keywords,
                suggested_tools=self._suggest_tools_for_intent(best_intent, entities),
                reasoning=f"LLM failed, using patterns: {str(e)[:50]}",
                ambiguous=True
            )
    
    def _suggest_tools_for_intent(self, intent_type: IntentType, entities: List[Entity]) -> List[str]:
        """Suggest tools based on intent and entities."""
        tool_mapping = {
            IntentType.FILE_READ: ['read_file'],
            IntentType.CODE_SEARCH: ['search_pattern', 'search_files'],
            IntentType.EXPLAIN: ['read_file', 'search_pattern'],
            IntentType.EXPLORE: ['list_directory'],
            IntentType.UNKNOWN: ['smart_suggest', 'fuzzy_find']
        }
        
        tools = tool_mapping.get(intent_type, ['smart_suggest'])
        
        # Add fuzzy find if no specific file
        if not any(e.type in [EntityType.FILE_PATH, EntityType.FILE_NAME] for e in entities):
            if 'fuzzy_find' not in tools and intent_type == IntentType.FILE_READ:
                tools.insert(0, 'fuzzy_find')
        
        return tools
    
    def _get_cache_key(self, query: str, context: Optional[Dict]) -> str:
        """Generate cache key."""
        # Create a stable hash
        key_parts = [query]
        if context:
            # Only include relevant context parts
            if 'current_file' in context:
                key_parts.append(f"file:{context['current_file']}")
            if 'current_directory' in context:
                key_parts.append(f"dir:{context['current_directory']}")
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_fallback_intent(self, query: str, reason: str) -> ProcessedIntent:
        """Get safe fallback intent."""
        return ProcessedIntent(
            type=IntentType.UNKNOWN,
            confidence=0.1,
            entities=[],
            keywords=[],
            suggested_tools=['smart_suggest', 'fuzzy_find'],
            reasoning=f"Fallback due to: {reason}",
            ambiguous=True
        )
    
    def _create_error_intent(self, error_type: str, message: str) -> ProcessedIntent:
        """Create intent for error cases."""
        return ProcessedIntent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            entities=[],
            keywords=[],
            suggested_tools=[],
            reasoning=f"Error: {error_type} - {message}",
            ambiguous=True
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.get_metrics()
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        return {
            'status': 'healthy' if self.circuit_breaker.state == CircuitBreakerState.CLOSED else 'degraded',
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'cache_size': len(self.cache.cache),
            'metrics': self.get_metrics()
        }