# LLM Module

A production-ready Python module for interacting with multiple Large Language Model providers with advanced features like structured output, streaming, middleware, and connection pooling.

## Features

### 🚀 Core Capabilities
- **Multi-Provider Support**: Seamlessly switch between Anthropic Claude and OpenAI GPT models
- **Structured Output**: Use Pydantic models with Instructor for type-safe responses
- **Streaming**: Real-time token streaming for responsive applications
- **Connection Pooling**: Efficient resource management for high-throughput scenarios
- **Async/Await**: Full async support for non-blocking operations

### 🛡️ Middleware System
- **Retry Logic**: Automatic retries with exponential backoff
- **Rate Limiting**: Prevent API throttling with configurable limits
- **Token Management**: Track and control token usage
- **Metrics Collection**: Monitor performance and usage statistics
- **Request/Response Logging**: Comprehensive logging for debugging

### 🎯 Advanced Features
- **Multi-turn Conversations**: Maintain context across interactions
- **Temperature Control**: Fine-tune response creativity
- **Custom Models**: Support for specific model versions
- **Error Handling**: Robust exception handling with detailed error messages
- **Provider Fallback**: Automatic fallback to alternative providers

## Installation

```bash
# Install required dependencies
pip install anthropic openai instructor pydantic tiktoken
```

## Quick Start

```python
import asyncio
from src.llm import LLMManager, Message, MessageRole, LLMProvider

async def main():
    # Initialize manager
    manager = LLMManager()
    
    # Get client for specific provider
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # Create messages
    messages = [
        Message(role=MessageRole.USER, content="What is Python?")
    ]
    
    # Get completion
    response = await client.complete(messages=messages)
    print(response.content)

asyncio.run(main())
```

## Structured Output Example

```python
from pydantic import BaseModel, Field
from typing import List

class CodeAnalysis(BaseModel):
    language: str = Field(description="Programming language")
    functions: List[str] = Field(description="Function names")
    complexity: str = Field(description="Code complexity level")
    suggestions: List[str] = Field(description="Improvement suggestions")

async def analyze_code():
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    code = "def hello(): print('Hello, World!')"
    
    messages = [
        Message(
            role=MessageRole.USER,
            content=f"Analyze this code: {code}"
        )
    ]
    
    # Get structured response
    analysis = await client.complete(
        messages=messages,
        response_model=CodeAnalysis
    )
    
    print(f"Language: {analysis.language}")
    print(f"Functions: {analysis.functions}")
```

## Streaming Example

```python
async def stream_story():
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a short story about AI"
        )
    ]
    
    # Stream response token by token
    async for chunk in client.stream(messages=messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
```

## Configuration

```python
from src.llm import LLMConfig

config = LLMConfig(
    # Retry settings
    retry_max_attempts=3,
    retry_initial_delay=1.0,
    retry_max_delay=10.0,
    
    # Rate limiting
    rate_limit_requests_per_minute=60,
    
    # Connection pool
    pool_size=10,
    pool_timeout=30.0,
    
    # Logging and metrics
    enable_logging=True,
    enable_metrics=True,
    
    # Token limits
    max_tokens_per_request=4000,
    max_tokens_per_minute=100000
)

manager = LLMManager(config=config)
```

## Middleware

The module includes several middleware components:

### Retry Middleware
Automatically retries failed requests with exponential backoff:
```python
config = LLMConfig(
    retry_max_attempts=3,
    retry_initial_delay=1.0,
    retry_exponential_base=2.0
)
```

### Rate Limiter
Prevents API throttling:
```python
config = LLMConfig(
    rate_limit_requests_per_minute=60,
    rate_limit_tokens_per_minute=100000
)
```

### Metrics Collector
Track usage and performance:
```python
manager = LLMManager(config=LLMConfig(enable_metrics=True))

# Get metrics
metrics = manager.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Average latency: {metrics['average_latency_ms']}ms")
```

## Error Handling

```python
from src.llm import LLMException, RateLimitError, AuthenticationError

try:
    response = await client.complete(messages=messages)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Wait and retry
except AuthenticationError as e:
    print(f"Auth failed: {e}")
    # Check API keys
except LLMException as e:
    print(f"LLM error: {e}")
    # Handle general errors
```

## Environment Variables

Set your API keys:
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Advanced Usage

### Multi-turn Conversations
```python
messages = []

# System message
messages.append(Message(
    role=MessageRole.SYSTEM,
    content="You are a helpful assistant"
))

# Conversation loop
for user_input in ["Hello", "What's Python?", "Show example"]:
    messages.append(Message(role=MessageRole.USER, content=user_input))
    
    response = await client.complete(messages=messages)
    print(f"Assistant: {response.content}")
    
    messages.append(Message(
        role=MessageRole.ASSISTANT,
        content=response.content
    ))
```

### Provider Fallback
```python
async def with_fallback():
    manager = LLMManager()
    
    providers = [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
    
    for provider in providers:
        try:
            client = manager.get_client(provider)
            response = await client.complete(messages=messages)
            return response
        except LLMException:
            continue
    
    raise Exception("All providers failed")
```

## Module Structure

```
src/llm/
├── __init__.py           # Main exports
├── core/
│   ├── client.py         # Base client implementation
│   ├── config.py         # Configuration classes
│   ├── exceptions.py     # Custom exceptions
│   ├── manager.py        # LLM manager
│   ├── pool.py          # Connection pooling
│   └── types.py         # Type definitions
├── middleware/
│   ├── base.py          # Middleware base class
│   ├── logging.py       # Logging middleware
│   ├── metrics.py       # Metrics collection
│   ├── rate_limiter.py  # Rate limiting
│   ├── retry.py         # Retry logic
│   └── token_calculator.py # Token counting
└── providers/
    ├── anthropic.py     # Anthropic Claude provider
    ├── base.py          # Provider base class
    └── openai.py        # OpenAI GPT provider
```

## Performance Tips

1. **Use Connection Pooling**: For high-throughput applications
2. **Enable Caching**: Cache frequent requests
3. **Batch Requests**: Process multiple items together
4. **Stream Long Responses**: For better UX
5. **Monitor Metrics**: Track performance and costs

## License

MIT License - See LICENSE file for details