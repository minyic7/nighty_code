"""
Example usage of the production-ready LLM module.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm import (
    get_llm_manager,
    Message,
    MessageRole,
    LLMProvider
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def basic_example():
    """Basic usage example."""
    # Get the LLM manager (singleton)
    manager = await get_llm_manager()
    
    # Get a client for the default provider
    client = manager.get_client()
    
    # Simple chat
    response = await client.chat(
        user_message="What is Python?",
        system_message="You are a helpful programming assistant. Keep responses concise."
    )
    
    print(f"Response: {response}")
    
    # Clean shutdown
    await manager.close()


async def advanced_example():
    """Advanced usage with multiple messages and streaming."""
    manager = await get_llm_manager()
    client = manager.get_client()
    
    # Build conversation history
    messages = [
        Message(MessageRole.SYSTEM, "You are a helpful assistant."),
        Message(MessageRole.USER, "What is machine learning?"),
        Message(MessageRole.ASSISTANT, "Machine learning is a subset of AI that enables systems to learn from data."),
        Message(MessageRole.USER, "Can you give me a simple example?")
    ]
    
    # Non-streaming response
    response = await client.complete(
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Response: {response.content}")
    
    if response.usage:
        print(f"Tokens used: {response.usage}")
    
    # Streaming response
    print("\nStreaming response:")
    async for chunk in client.stream_complete(messages):
        print(chunk, end="", flush=True)
    print()
    
    await manager.close()


async def multi_provider_example():
    """Example using multiple providers."""
    manager = await get_llm_manager()
    
    # Check available providers
    providers = manager.list_providers()
    print(f"Available providers: {providers}")
    
    # Get provider status
    for provider in providers:
        status = manager.get_status(provider)
        print(f"\n{provider} status:")
        print(f"  Total configs: {status.get('total_configs', 0)}")
        print(f"  Total clients: {status.get('total_clients', 0)}")
        print(f"  Available: {status.get('available_clients', 0)}")
    
    # Use specific provider if available
    if LLMProvider.ANTHROPIC in providers:
        client = manager.get_client(LLMProvider.ANTHROPIC)
        response = await client.chat("Hello! How are you?")
        print(f"\nAnthropic response: {response}")
    
    await manager.close()


async def error_handling_example():
    """Example showing error handling."""
    from src.llm.core.exceptions import (
        LLMException,
        LLMRateLimitError,
        LLMAuthenticationError
    )
    
    manager = await get_llm_manager()
    client = manager.get_client()
    
    try:
        # This might fail due to rate limits or other issues
        response = await client.chat(
            user_message="Test message",
            max_tokens=10
        )
        print(f"Success: {response}")
        
    except LLMRateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        # Could implement backoff or switch providers
        
    except LLMAuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        # Need to check API keys
        
    except LLMException as e:
        logger.error(f"LLM error: {e}")
        # General LLM error
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        await manager.close()


async def metrics_example():
    """Example showing metrics collection."""
    manager = await get_llm_manager()
    client = manager.get_client()
    
    # Make several requests
    for i in range(5):
        try:
            response = await client.chat(
                user_message=f"Count to {i+1}",
                max_tokens=50
            )
            print(f"Request {i+1} completed")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    # Get pool status (includes metrics)
    status = manager.get_status()
    print("\nPool metrics:")
    for provider, provider_status in status.items():
        print(f"\n{provider}:")
        metrics = provider_status.get('metrics', {})
        print(f"  Total requests: {metrics.get('total_requests', 0)}")
        print(f"  Failed requests: {metrics.get('failed_requests', 0)}")
        print(f"  Average wait time: {metrics.get('average_wait_time', 0):.2f}s")
    
    await manager.close()


async def main():
    """Run examples."""
    print("=" * 50)
    print("Basic Example")
    print("=" * 50)
    await basic_example()
    
    print("\n" + "=" * 50)
    print("Advanced Example")
    print("=" * 50)
    await advanced_example()
    
    print("\n" + "=" * 50)
    print("Multi-Provider Example")
    print("=" * 50)
    await multi_provider_example()
    
    print("\n" + "=" * 50)
    print("Error Handling Example")
    print("=" * 50)
    await error_handling_example()
    
    print("\n" + "=" * 50)
    print("Metrics Example")
    print("=" * 50)
    await metrics_example()


if __name__ == "__main__":
    asyncio.run(main())