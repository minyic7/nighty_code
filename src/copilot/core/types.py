# src/copilot/core/types.py
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ThoughtType(str, Enum):
    """Types of thoughts in the cognitive flow"""
    UNDERSTANDING = "understanding"
    HYPOTHESIS = "hypothesis"
    PLANNING = "planning"
    REASONING = "reasoning"
    REFLECTION = "reflection"
    DECISION = "decision"
    FAST_RESPONSE = "fast_response"


class ActionType(str, Enum):
    """Types of actions the copilot can take"""
    TOOL_USE = "tool_use"
    QUERY_USER = "query_user"
    PROVIDE_ANSWER = "provide_answer"
    REQUEST_CLARIFICATION = "request_clarification"
    ERROR_RECOVERY = "error_recovery"


class ToolStatus(str, Enum):
    """Status of tool execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Hypothesis(BaseModel):
    """Represents a hypothesis about user intent"""
    intent: str = Field(description="The hypothesized intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this hypothesis")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    test_cost: float = Field(default=1.0, description="Cost to test this hypothesis (0=cheap, 1=expensive)")
    tested: bool = Field(default=False, description="Whether this hypothesis has been tested")
    test_result: Optional[float] = Field(default=None, description="Confidence after testing")


class ThoughtProcess(BaseModel):
    """Represents a single thought in the cognitive flow"""
    type: ThoughtType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hypotheses: List[Hypothesis] = Field(default_factory=list, description="Multiple hypotheses if applicable")


class ToolCall(BaseModel):
    """Represents a call to an MCP tool"""
    tool_name: str
    server_name: str  # Which MCP server provides this tool
    arguments: Dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ActionPlan(BaseModel):
    """Represents a plan of action"""
    goal: str
    steps: List[str]
    current_step: int = 0
    tool_calls: List[ToolCall] = Field(default_factory=list)
    status: Literal["planning", "executing", "completed", "failed"] = "planning"
    reasoning: str
    alternatives: List[str] = Field(default_factory=list)


class Memory(BaseModel):
    """Represents a memory item"""
    content: str
    type: Literal["conversation", "fact", "tool_result", "error", "learning"]
    importance: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserMessage(BaseModel):
    """User input message"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None


class AssistantResponse(BaseModel):
    """Assistant response"""
    content: str
    thoughts: List[ThoughtProcess] = Field(default_factory=list)
    actions_taken: List[ToolCall] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


MAX_HISTORY_SIZE = 1000  # Limit history to prevent unbounded growth

class ConfidenceScores(BaseModel):
    """Track confidence throughout the pipeline"""
    understanding: float = Field(default=0.0, ge=0.0, le=1.0)
    planning: float = Field(default=0.0, ge=0.0, le=1.0)
    execution: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def update_overall(self):
        """Calculate overall confidence as weighted average"""
        scores = [self.understanding, self.planning, self.execution]
        valid_scores = [s for s in scores if s > 0]
        self.overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        return self.overall
    
    def should_clarify(self) -> bool:
        """Return True if confidence is too low and clarification needed"""
        return self.overall < 0.5
    
    def should_show_alternatives(self) -> bool:
        """Return True if confidence is moderate and alternatives should be shown"""
        return 0.5 <= self.overall < 0.7


class CopilotState(BaseModel):
    """Complete state of the copilot at any point"""
    # Conversation
    messages: List[UserMessage | AssistantResponse] = Field(default_factory=list)
    
    # Cognitive state
    current_thought: Optional[ThoughtProcess] = None
    thought_history: List[ThoughtProcess] = Field(default_factory=list)
    current_hypotheses: List[Hypothesis] = Field(default_factory=list)
    
    # Planning state
    current_plan: Optional[ActionPlan] = None
    plan_history: List[ActionPlan] = Field(default_factory=list)
    
    # Memory
    short_term_memory: List[Memory] = Field(default_factory=list)
    long_term_memory: List[Memory] = Field(default_factory=list)
    
    # Tool state
    available_tools: Dict[str, List[str]] = Field(default_factory=dict)  # server_name -> tool_names
    tool_history: List[ToolCall] = Field(default_factory=list)
    
    # Confidence tracking
    confidence_scores: ConfidenceScores = Field(default_factory=ConfidenceScores)
    
    # Fast path cache
    fast_path_patterns: Dict[str, str] = Field(default_factory=dict)  # pattern -> response
    
    # Session metadata
    session_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    # Configuration
    max_retries: int = 3
    confidence_threshold: float = 0.7
    memory_limit: int = 100  # Max items in short-term memory
    
    def add_thought(self, thought: ThoughtProcess):
        """Add a thought to the history"""
        self.current_thought = thought
        self.thought_history.append(thought)
        # Limit history size to prevent unbounded growth
        if len(self.thought_history) > MAX_HISTORY_SIZE:
            self.thought_history = self.thought_history[-MAX_HISTORY_SIZE:]
        self.last_activity = datetime.now()
    
    def add_memory(self, memory: Memory, long_term: bool = False):
        """Add a memory item"""
        if long_term:
            self.long_term_memory.append(memory)
        else:
            self.short_term_memory.append(memory)
            # Manage memory limit
            if len(self.short_term_memory) > self.memory_limit:
                # Check ALL items for importance, not just first 10
                important_memories = [m for m in self.short_term_memory if m.importance > 0.7]
                
                # Avoid duplicates in long-term memory
                for memory in important_memories:
                    if memory not in self.long_term_memory:
                        self.long_term_memory.append(memory)
                
                # Keep recent non-important memories
                remaining = [m for m in self.short_term_memory if m not in important_memories]
                self.short_term_memory = remaining[-self.memory_limit:]
        self.last_activity = datetime.now()
    
    def update_plan(self, plan: ActionPlan):
        """Update the current plan"""
        if self.current_plan:
            self.plan_history.append(self.current_plan)
            # Limit history size
            if len(self.plan_history) > MAX_HISTORY_SIZE:
                self.plan_history = self.plan_history[-MAX_HISTORY_SIZE:]
        self.current_plan = plan
        self.last_activity = datetime.now()
    
    def record_tool_call(self, tool_call: ToolCall):
        """Record a tool call"""
        self.tool_history.append(tool_call)
        # Limit history size
        if len(self.tool_history) > MAX_HISTORY_SIZE:
            self.tool_history = self.tool_history[-MAX_HISTORY_SIZE:]
        if self.current_plan:
            self.current_plan.tool_calls.append(tool_call)
        self.last_activity = datetime.now()


class GuardrailCheck(BaseModel):
    """Result of a guardrail check"""
    passed: bool
    check_type: str
    message: Optional[str] = None
    severity: Literal["info", "warning", "error"] = "info"
    suggestions: List[str] = Field(default_factory=list)


class CopilotConfig(BaseModel):
    """Configuration for the copilot"""
    # LLM settings
    llm_provider: str = "anthropic"
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Behavior settings
    verbose: bool = False
    auto_execute_tools: bool = True
    require_confirmation: bool = False
    
    # Memory settings
    enable_memory: bool = True
    memory_persistence: bool = False
    memory_file: Optional[str] = None
    
    # Guardrails
    enable_guardrails: bool = True
    safety_checks: bool = True
    
    # Tool settings
    tool_timeout: int = 30  # seconds
    max_parallel_tools: int = 3