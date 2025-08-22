# LLM Module

High-performance LLM client with connection pooling, rate limiting, and middleware support.

## Quick Start

```python
from llm import get_llm_manager, LLMProvider, Message, MessageRole

async def main():
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [Message(MessageRole.USER, "Hello, world!")]
    response = await client.complete(messages)
    print(response.content)
    
    await manager.close()
```

## Features

- **Connection Pooling**: Multiple API keys with automatic load balancing
- **Rate Limiting**: Token-accurate limiting using native provider APIs  
- **Middleware Stack**: Logging, metrics, retry, and rate limit middleware
- **Async Support**: Built on asyncio for maximum concurrency
- **Multi-Provider**: Unified interface for Anthropic, OpenAI, etc.

## Configuration

Edit `config/llm.yaml`:

```yaml
providers:
  anthropic:
    api_keys: [key1, key2, ...]  # Multiple keys for load balancing
    models: [claude-3-5-haiku-20241022]
    rate_limits:
      - requests_per_minute: 12
        input_tokens_per_minute: 5000
        output_tokens_per_minute: 2000

pool:
  max_size: 4  # Match number of API keys
  acquire_timeout: 30.0
```

## Basic Use Cases

### Simple Completion
```python
messages = [Message(MessageRole.USER, "Explain quantum computing")]
response = await client.complete(messages, max_tokens=200)
```

### Streaming Response
```python
messages = [Message(MessageRole.USER, "Write a story")]
async for chunk in client.stream_complete(messages):
    print(chunk, end='', flush=True)
```

### Conversation with History
```python
from llm import SessionManager

session = SessionManager()
session.add_message(MessageRole.USER, "My name is Alice")
response = await client.complete(session.get_messages())
session.add_message(MessageRole.ASSISTANT, response.content)

# Later in conversation
session.add_message(MessageRole.USER, "What's my name?")
response = await client.complete(session.get_messages())
# Response will remember "Alice"
```

## Advanced Use Cases

### Custom System Prompts
```python
messages = [
    Message(MessageRole.SYSTEM, "You are a helpful coding assistant"),
    Message(MessageRole.USER, "Write a Python function to sort a list")
]
response = await client.complete(messages)
```

### Parallel Processing with Pool
```python
async def process_batch(prompts):
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    tasks = []
    for prompt in prompts:
        messages = [Message(MessageRole.USER, prompt)]
        tasks.append(client.complete(messages))
    
    results = await asyncio.gather(*tasks)
    return [r.content for r in results]

# Process 50 prompts in parallel
prompts = [f"Question {i}" for i in range(50)]
answers = await process_batch(prompts)
```

### Monitor Pool Status
```python
status = manager.get_status(LLMProvider.ANTHROPIC)
print(f"Available clients: {status['available_clients']}/{status['total_clients']}")
print(f"Queue size: {status['queue_size']}")

# Check individual client status
for client_status in status['client_statuses']:
    if client_status['rate_limit_status']:
        print(f"Client {client_status['client_id']}:")
        print(f"  Tokens remaining: {client_status['rate_limit_status']['tokens_remaining']}")
```

### Rate Limit Handling
```python
from llm import LLMRateLimitError

try:
    response = await client.complete(messages)
except LLMRateLimitError as e:
    print(f"Rate limited: {e}")
    # The pool automatically queues requests when rate limited
    # No manual retry needed
```

### Custom Middleware
```python
from llm.middleware.base import Middleware

class CustomMiddleware(Middleware):
    async def process_request(self, request, context):
        # Add custom headers, logging, etc.
        context['custom_data'] = 'value'
        return request
    
    async def process_response(self, response, context):
        # Process response
        response.metadata['custom'] = context.get('custom_data')
        return response

# Add to provider
provider.middleware_chain.add(CustomMiddleware())
```

### Load Testing
```python
async def load_test(num_requests=100):
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    start = time.time()
    tasks = []
    
    for i in range(num_requests):
        messages = [Message(MessageRole.USER, f"Test {i}")]
        tasks.append(client.complete(messages, max_tokens=10))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success = sum(1 for r in results if not isinstance(r, Exception))
    duration = time.time() - start
    
    print(f"Completed {success}/{num_requests} in {duration:.2f}s")
    print(f"Throughput: {success/duration:.2f} req/s")
```

### Export Metrics
```python
# After running requests
pool = manager._pools[LLMProvider.ANTHROPIC]
for client in pool._clients:
    metrics = client.middleware_chain.get_middleware(MetricsMiddleware).get_metrics()
    print(f"Client metrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Avg latency: {metrics['avg_latency']:.2f}ms")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
```

## Architecture

```
Application → LLMManager → ConnectionPool → Middleware → Provider
                  ↓             ↓               ↓
              (Singleton)  (Load Balance)  (Log/Retry/Rate)
```

## Middleware Stack

1. **LoggingMiddleware**: Request/response logging
2. **MetricsMiddleware**: Performance tracking  
3. **RetryMiddleware**: Exponential backoff for failures
4. **RateLimitMiddleware**: Token-accurate rate limiting

## Error Handling

- `LLMRateLimitError`: Rate limit exceeded (auto-queued)
- `LLMPoolExhaustedError`: No clients available  
- `LLMTimeoutError`: Request timeout
- `LLMAuthenticationError`: Invalid API key
- `LLMProviderError`: Provider-specific errors

## Performance Tips

1. Use multiple API keys for higher throughput
2. Set `pool.max_size` = number of API keys
3. Adjust `acquire_timeout` based on load
4. Enable connection reuse with proper `idle_timeout`
5. Monitor metrics to optimize rate limits

## Testing

```bash
# Test basic functionality
python test_llm.py

# Interactive session
python interactive_llm.py

# Load testing
python test_pool_load.py
```

## License

MIT