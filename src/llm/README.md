# LLM Module

A robust, production-ready LLM client module with built-in connection pooling, automatic retries, and Instructor integration for structured outputs.

## Features

### Core Capabilities
- **Multi-Provider Support**: Anthropic, OpenAI, Google, Groq, and local models
- **Connection Pooling**: Efficient management of multiple API keys with load balancing
- **Automatic Retries**: Exponential backoff with jitter for transient failures
- **Structured Outputs**: Seamless Instructor integration for type-safe responses
- **Rate Limiting**: Built-in rate limit handling and backpressure management
- **Unified Interface**: Consistent API across all providers

### Key Components

#### 1. LLM Manager (`core/manager.py`)
Centralized management of LLM clients:
- Provider registration and configuration
- Client lifecycle management
- Configuration validation
- Singleton pattern for resource efficiency

#### 2. Base Provider (`providers/base.py`)
Abstract base for all LLM providers:
- Common interface definition
- Instructor integration
- Middleware support
- Error handling patterns

#### 3. Connection Pool (`core/pool.py`)
Advanced connection management:
- Round-robin load balancing across API keys
- Health tracking per connection
- Automatic failover on errors
- Configurable pool sizes

## Architecture

```
LLMManager (Singleton)
    ↓
Provider Registry
    ↓
Connection Pool (per provider)
    ↓
Individual Clients (with Instructor)
    ↓
API Endpoints
```

## Installation

```bash
pip install instructor  # For structured outputs
pip install anthropic  # For Anthropic
pip install openai     # For OpenAI
```

## Usage

### Basic Usage

```python
from src.llm import get_llm_manager, LLMProvider, Message, MessageRole

# Get manager instance
manager = await get_llm_manager()

# Get a client
client = manager.get_client(LLMProvider.ANTHROPIC)

# Make a completion request
messages = [
    Message(MessageRole.SYSTEM, "You are a helpful assistant"),
    Message(MessageRole.USER, "What is the capital of France?")
]

response = await client.complete(messages)
print(response.content)
```

### Structured Outputs with Instructor

```python
from pydantic import BaseModel, Field
from typing import List

# Define your output schema
class CityInfo(BaseModel):
    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Population count")
    landmarks: List[str] = Field(description="Famous landmarks")

# Request structured output
city_info = await client.complete(
    messages=[
        Message(MessageRole.USER, "Tell me about Paris")
    ],
    response_model=CityInfo  # Auto-detects and uses Instructor
)

print(f"City: {city_info.name}")
print(f"Population: {city_info.population:,}")
print(f"Landmarks: {', '.join(city_info.landmarks)}")
```

### Connection Pooling

```python
# Configure multiple API keys for load balancing
import os

# Set multiple keys (comma-separated)
os.environ['ANTHROPIC_API_KEYS'] = 'key1,key2,key3'

# The pool automatically rotates between keys
manager = await get_llm_manager()
client = manager.get_client(LLMProvider.ANTHROPIC)

# Make many requests - automatically distributed
for i in range(100):
    response = await client.complete(messages)
```

### Error Handling

```python
from src.llm.core.exceptions import (
    LLMError,
    RateLimitError,
    InvalidRequestError
)

try:
    response = await client.complete(messages)
except RateLimitError as e:
    print(f"Rate limited: {e}. Retry after: {e.retry_after}")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except LLMError as e:
    print(f"General LLM error: {e}")
```

### Advanced Configuration

```python
from src.llm import LLMConfig

config = LLMConfig(
    temperature=0.7,
    max_tokens=2000,
    max_retries=3,
    retry_delay=1.0,
    timeout=30.0,
    system_prompt="You are a technical assistant"
)

response = await client.complete(
    messages,
    **config.to_dict()
)
```

## Provider-Specific Features

### Anthropic
- Claude 3 models (Opus, Sonnet, Haiku)
- 200k context window
- System prompts support
- Vision capabilities

### OpenAI
- GPT-4 and GPT-3.5 models
- Function calling
- JSON mode
- Vision capabilities

### Configuration

Set API keys via environment variables:

```bash
# Single key
export ANTHROPIC_API_KEY="your-key"

# Multiple keys for pooling
export ANTHROPIC_API_KEYS="key1,key2,key3"

# Other providers
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"
```

## Instructor Integration

The module automatically detects when a `response_model` is provided and uses Instructor for structured outputs:

### How It Works

1. **Auto-Detection**: When `response_model` parameter is present, Instructor is activated
2. **Type Safety**: Responses are validated against your Pydantic models
3. **Retry Logic**: Automatic retries on validation failures
4. **Error Handling**: Clear error messages for schema mismatches

### Example: Complex Extraction

```python
from pydantic import BaseModel
from typing import List, Optional

class CodeFunction(BaseModel):
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]

class CodeAnalysis(BaseModel):
    language: str
    functions: List[CodeFunction]
    imports: List[str]
    complexity: int = Field(ge=1, le=10)

# Analyze code with structured output
analysis = await client.complete(
    messages=[
        Message(MessageRole.USER, f"Analyze this code:\n{code_snippet}")
    ],
    response_model=CodeAnalysis
)

# Type-safe access to results
for func in analysis.functions:
    print(f"Function: {func.name}({', '.join(func.parameters)})")
```

## Connection Pool Details

### Load Balancing Strategy
- **Round-Robin**: Distributes requests evenly across connections
- **Health Tracking**: Marks connections as unhealthy on errors
- **Automatic Recovery**: Retries unhealthy connections after cooldown

### Pool Configuration
```python
# Default pool configuration
pool = ConnectionPool(
    api_keys=['key1', 'key2'],
    max_connections_per_key=5,  # Concurrent requests per key
    health_check_interval=60,    # Seconds between health checks
)
```

## Error Handling

### Retry Strategy
- Exponential backoff with jitter
- Configurable max retries (default: 3)
- Different strategies for different error types:
  - Rate limits: Respect retry-after headers
  - Network errors: Immediate retry with backoff
  - Invalid requests: No retry

### Error Types
- `LLMError`: Base exception for all LLM errors
- `RateLimitError`: Rate limit exceeded
- `InvalidRequestError`: Malformed request
- `AuthenticationError`: Invalid API key
- `NetworkError`: Connection issues

## Performance Tips

1. **Use Connection Pooling**: Distribute load across multiple API keys
2. **Enable Caching**: Cache frequently used completions
3. **Batch Requests**: Group related requests when possible
4. **Set Appropriate Timeouts**: Avoid hanging on slow requests
5. **Use Structured Outputs**: More efficient than parsing text

## Testing

```python
# Test with mock client
from src.llm.testing import MockLLMClient

mock_client = MockLLMClient(
    responses=["Response 1", "Response 2"]
)

response = await mock_client.complete(messages)
assert response.content == "Response 1"
```

## Best Practices

1. **Always Use Type Hints**: Enables better IDE support and validation
2. **Handle Errors Gracefully**: Implement proper error handling
3. **Monitor Usage**: Track token usage and costs
4. **Use Structured Outputs**: When you need specific data formats
5. **Set Temperature Appropriately**: Lower for factual, higher for creative

## Future Enhancements

- [ ] Streaming support for long responses
- [ ] Token usage tracking and cost estimation
- [ ] Response caching with Redis backend
- [ ] Prompt template management
- [ ] Fine-tuning integration
- [ ] Multi-modal support (images, audio)

## License

See main project LICENSE file.