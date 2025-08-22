# LLM Module - Production-Ready Architecture

## Overview

A production-ready LLM (Large Language Model) module with connection pooling, rate limiting, middleware support, and multi-provider compatibility.

## Features

### Core Features
- **Connection Pooling**: Efficient resource management with configurable pool sizes
- **Rate Limiting**: Sliding window rate limiting with RPM/TPM tracking
- **Multi-Provider Support**: OpenAI and Anthropic with extensible architecture
- **Load Balancing**: Round-robin distribution across multiple API keys
- **Middleware Pipeline**: Pluggable middleware for cross-cutting concerns
- **Automatic Retries**: Exponential backoff with jitter for transient failures
- **Comprehensive Logging**: Request/response tracking with performance metrics
- **Metrics Collection**: Detailed metrics for monitoring and analysis

### Architecture

```
src/llm/
├── core/                 # Core components
│   ├── client.py        # High-level client interface
│   ├── pool.py          # Connection pool management
│   ├── manager.py       # Singleton manager for all operations
│   ├── config.py        # Configuration management
│   ├── types.py         # Type definitions
│   └── exceptions.py    # Custom exceptions
├── providers/           # Provider implementations
│   ├── base.py         # Base provider interface
│   ├── anthropic.py    # Anthropic/Claude implementation
│   └── openai.py       # OpenAI implementation
├── middleware/          # Middleware components
│   ├── base.py         # Middleware interface
│   ├── rate_limiter.py # Rate limiting middleware
│   ├── retry.py        # Retry logic middleware
│   ├── logging.py      # Logging middleware
│   └── metrics.py      # Metrics collection middleware
└── utils/              # Utility functions
```

## Configuration

Configuration is managed via `config/llm.yaml`:

```yaml
providers:
  anthropic:
    api_keys:
      - key1
      - key2
      - key3
    models:
      - claude-3-5-haiku-20241022
      - claude-3-5-sonnet-20241022
      - claude-3-opus-20240229
    rate_limits:
      - requests_per_minute: 50
        input_tokens_per_minute: 50000
        output_tokens_per_minute: 10000
```

### Key Configuration Concepts

1. **Multiple API Keys**: Each provider can have multiple API keys for load balancing
2. **Per-Key Models**: Each API key can use a different model
3. **Per-Key Rate Limits**: Each API key can have different rate limits
4. **Validation**: Number of api_keys = models = rate_limits (if using list)

## Usage

### Basic Usage

```python
from src.llm import get_llm_manager

# Get the global manager
manager = await get_llm_manager()

# Get a client
client = manager.get_client()

# Simple chat
response = await client.chat("Hello, how are you?")
print(response)
```

### Advanced Usage

```python
from src.llm import Message, MessageRole

# Build conversation
messages = [
    Message(MessageRole.SYSTEM, "You are a helpful assistant"),
    Message(MessageRole.USER, "What is Python?")
]

# Get completion with options
response = await client.complete(
    messages=messages,
    temperature=0.7,
    max_tokens=200
)

# Streaming
async for chunk in client.stream_complete(messages):
    print(chunk, end="")
```

## Middleware System

The middleware system provides a clean way to add cross-cutting concerns:

### Available Middleware

1. **RateLimitMiddleware**: Enforces rate limits with sliding window tracking
2. **RetryMiddleware**: Handles transient failures with exponential backoff
3. **LoggingMiddleware**: Logs requests, responses, and errors
4. **MetricsMiddleware**: Collects detailed metrics for monitoring

### Middleware Pipeline

```
Request → RateLimit → Retry → Logging → Metrics → Provider
                                                       ↓
Response ← Metrics ← Logging ← Retry ← RateLimit ← Provider
```

## Rate Limiting

### How It Works

- **Sliding Window**: Each request has its own 60-second window
- **Independent Tracking**: RPM and TPM tracked separately
- **Gradual Recovery**: Capacity recovers as old requests expire
- **Per-Client Limits**: Each API key has independent rate limits

### Rate Limit Configuration

```yaml
rate_limits:
  requests_per_minute: 50
  input_tokens_per_minute: 50000
  output_tokens_per_minute: 10000
```

## Load Balancing

The pool creates clients in round-robin fashion:
- Client 1 → API key 1, Model 1, Rate limits 1
- Client 2 → API key 2, Model 2, Rate limits 2
- Client 3 → API key 3, Model 3, Rate limits 3
- Client 4 → API key 1 (cycles back)

## Error Handling

### Exception Hierarchy

```python
LLMException (base)
├── LLMProviderError      # Provider-specific errors
├── LLMConnectionError    # Connection failures
├── LLMRateLimitError     # Rate limit exceeded
├── LLMAuthenticationError # Auth failures
├── LLMTimeoutError       # Request timeouts
├── LLMPoolExhaustedError # No available clients
└── LLMConfigurationError # Config issues
```

### Error Recovery

- **Automatic Retries**: Transient errors retry with backoff
- **Provider Fallback**: Can fallback to other providers (future)
- **Circuit Breaking**: Prevent cascading failures (future)

## Monitoring & Observability

### Metrics Collection

```python
# Get metrics
status = manager.get_status()
print(status['metrics'])

# Metrics include:
# - Total/successful/failed requests
# - Token usage (input/output/total)
# - Latency (min/max/average)
# - Errors by type
# - Usage by model
```

### Logging

Comprehensive logging at multiple levels:
- Request/response details
- Performance metrics
- Error tracking
- Retry attempts

## Production Considerations

### Security
- API keys stored securely in config
- No secrets in logs (content logging disabled by default)
- Rate limiting prevents abuse

### Performance
- Connection pooling reduces overhead
- Async/await for non-blocking operations
- Efficient token estimation
- Minimal memory footprint

### Reliability
- Health checks for connection validity
- Automatic client recycling
- Graceful degradation
- Comprehensive error handling

### Scalability
- Multiple API keys for higher throughput
- Configurable pool sizes
- Per-provider scaling
- Middleware pipeline for extensions

## Future Enhancements

- [ ] Circuit breaker pattern
- [ ] OpenTelemetry integration
- [ ] Prometheus metrics export
- [ ] Response caching
- [ ] Token budget management
- [ ] Provider fallback chains
- [ ] Request prioritization
- [ ] WebSocket streaming support

## Testing

Run the examples to test the module:

```bash
python examples/llm_usage.py
```

## Best Practices

1. **Always use context managers** for proper cleanup
2. **Configure appropriate rate limits** based on your tier
3. **Monitor metrics** in production
4. **Use multiple API keys** for better throughput
5. **Enable retry middleware** for resilience
6. **Set reasonable timeouts** to prevent hanging
7. **Log errors** for debugging
8. **Rotate API keys** if one gets rate limited