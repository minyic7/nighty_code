# Copilot Module

An intelligent AI assistant module built with LangGraph, featuring multi-hypothesis thinking, confidence tracking, and seamless tool integration through MCP (Model Context Protocol).

## Features

### Core Capabilities
- **Structured LLM Outputs**: Leverages Instructor for type-safe, validated responses
- **Multi-Hypothesis Thinking**: Generates 3-5 interpretations for each user request
- **Dual-Speed Processing**: Fast path for simple queries, slow path for complex reasoning
- **Confidence Tracking**: Three-tier confidence system guides decision-making
- **Memory Management**: Smart short-term and long-term memory with importance-based preservation
- **Tool Integration**: Seamless MCP tool execution with intelligent routing
- **Guardrails**: Built-in safety checks and validation

### Key Components

#### 1. Cognitive Nodes (`nodes/cognitive.py`)
Handles the thinking process with **structured outputs via Instructor**:
- **Understanding**: Analyzes user intent with multiple hypotheses (uses `MultiHypothesisOutput`)
- **Planning**: Creates actionable plans based on understanding (uses `PlanningOutput`)
- **Reasoning**: Decides on next steps and adjusts plans (uses `ReasoningOutput`)
- **Reflection**: Learns from failures and improves responses (uses `ReflectionOutput`)

#### 2. Workflow Orchestration (`core/workflow.py`)
LangGraph-based workflow with:
- State management across conversation turns
- Intelligent routing between cognitive nodes
- Session persistence with checkpointing
- Configurable recursion limits

#### 3. Tool Router (`tools/router.py`)
MCP-based tool execution with **Instructor-powered selection**:
- Dynamic tool discovery from MCP servers
- LLM-guided tool selection (uses `ToolSelectionOutput` and `ToolExecutionPlan`)
- Argument normalization and validation
- Error handling with graceful degradation

#### 4. Memory Manager (`memory/manager.py`)
Sophisticated memory system:
- Importance-based memory preservation
- Automatic migration to long-term storage
- Duplicate prevention
- Bounded history with smart truncation

## Architecture

```
User Input
    ↓
[Entry Node]
    ↓
[Understanding] ← Multi-hypothesis generation
    ↓
[Planning] ← Create action plans
    ↓
[Reasoning] ← Decision making
    ↓
[Tool Execution] ← MCP tools
    ↓
[Response Generation]
    ↓
User Output
```

## Usage

### Basic Usage

```python
from src.copilot import CopilotWorkflow

# Initialize workflow
workflow = CopilotWorkflow()
await workflow.initialize()

# Process a message
response = await workflow.process_message(
    "Help me find Python files", 
    session_id="user-123"
)

print(f"Response: {response.content}")
print(f"Confidence: {response.confidence}")
```

### Configuration

```python
from src.copilot import CopilotConfig, CopilotWorkflow

config = CopilotConfig(
    llm_provider="anthropic",
    temperature=0.7,
    enable_memory=True,
    enable_guardrails=True,
    memory_persistence=False,
    auto_execute_tools=True
)

workflow = CopilotWorkflow(config=config)
```

### Interactive REPL

```python
from src.copilot.client import InteractiveCopilot

# Start interactive session
copilot = InteractiveCopilot()
await copilot.start()
```

## Instructor Integration

The copilot module extensively uses **Instructor** for structured, type-safe LLM outputs:

### Structured Output Models

```python
# Multi-hypothesis generation
class MultiHypothesisOutput(BaseModel):
    hypotheses: List[Hypothesis]  # 3-5 different interpretations
    primary_intent: str
    confidence: float
    
# Planning output
class PlanningOutput(BaseModel):
    goal: str
    steps: List[str]
    tool_requirements: List[str]
    estimated_complexity: float
    
# Tool selection
class ToolSelectionOutput(BaseModel):
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    reasoning: str
```

### Benefits
- **Type Safety**: All LLM responses are validated against Pydantic models
- **Consistency**: Structured outputs ensure consistent response formats
- **Error Handling**: Automatic validation and retry on schema mismatches
- **Better Reasoning**: Forces LLM to structure its thinking process

## Confidence System

The module uses a three-tier confidence system:

| Confidence | Action |
|------------|--------|
| < 50% | Request clarification |
| 50-70% | Offer alternatives |
| > 70% | Proceed with action |

## Memory Management

### Short-term Memory
- Limited to configurable size (default: 100 items)
- Stores recent interactions and facts
- Automatically migrates important items to long-term

### Long-term Memory
- Preserves important memories (importance > 0.7)
- No duplicates allowed
- Persistent across sessions (when enabled)

## Bug Fixes Applied

Recent critical fixes (all verified with tests):

1. **Memory Management**: Fixed bug where only first 10 items were checked for importance
2. **History Limits**: Added MAX_HISTORY_SIZE to prevent unbounded growth
3. **Input Validation**: Added smart truncation preserving start, middle, and end
4. **Tool Failure Handling**: Plans now stop on critical tool failures
5. **Fast Path Detection**: Fixed whitespace handling in pattern matching
6. **Hypothesis Updates**: Test results now properly update hypothesis list

## Testing

### Run Tests

```bash
# Unit tests
python test_copilot_integration.py

# Pressure tests
python test_copilot_final.py

# Memory tests
python test_memory_simple.py
```

### Test Coverage

- ✅ Memory management with overflow
- ✅ History size limits
- ✅ Input validation (empty, whitespace, long messages)
- ✅ Multi-turn conversations
- ✅ Tool execution and failure handling
- ✅ Edge cases (unicode, special characters)
- ✅ Confidence scoring and routing

## Performance

- Handles 100+ message conversations
- Memory-efficient with bounded data structures
- Graceful degradation under load
- Smart truncation for long inputs
- Parallel tool execution support

## Dependencies

- `langgraph`: Workflow orchestration
- `pydantic`: Data validation and structured outputs
- `instructor`: Type-safe LLM responses
- `src.llm`: LLM client management (with Instructor integration)
- `src.mcp`: Tool protocol implementation

## Production Readiness

✅ **Production Ready** - All critical bugs fixed and tested

### Monitoring Recommendations
- Track memory usage and history sizes
- Monitor response times and confidence scores
- Log tool execution failures
- Watch for recursion limit hits

### Scaling Considerations
- Add rate limiting for high-traffic scenarios
- Implement caching for repeated queries
- Consider distributed memory storage
- Add health check endpoints

## Future Improvements

- [ ] Parallel tool execution optimization
- [ ] Advanced caching layer
- [ ] Distributed memory backend
- [ ] Enhanced hypothesis testing
- [ ] Real-time learning from interactions
- [ ] Custom tool development SDK

## License

See main project LICENSE file.