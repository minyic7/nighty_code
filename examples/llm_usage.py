#!/usr/bin/env python3
"""
LLM Module Usage Examples

This script demonstrates various features of the LLM module including:
- Multiple provider support (Anthropic, OpenAI)
- Structured output with Pydantic models
- Streaming responses
- Middleware (retry, rate limiting, logging)
- Connection pooling
- Token management
"""

import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Set up environment variables (replace with your actual keys)
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")

from src.llm import (
    LLMManager,
    Message,
    MessageRole,
    LLMProvider,
    LLMConfig,
    LLMException
)

# Example 1: Basic text completion
async def basic_completion_example():
    """Simple text completion with different providers"""
    print("\n=== Basic Completion Example ===")
    
    manager = LLMManager()
    
    # Example with Anthropic
    try:
        client = manager.get_client(LLMProvider.ANTHROPIC)
        
        messages = [
            Message(role=MessageRole.USER, content="What is the capital of France?")
        ]
        
        response = await client.complete(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"Anthropic Response: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
    except LLMException as e:
        print(f"Anthropic error: {e}")
    
    # Example with OpenAI
    try:
        client = manager.get_client(LLMProvider.OPENAI)
        
        messages = [
            Message(role=MessageRole.USER, content="What is 2+2?")
        ]
        
        response = await client.complete(
            messages=messages,
            temperature=0
        )
        
        print(f"OpenAI Response: {response.content}")
    except LLMException as e:
        print(f"OpenAI error: {e}")


# Example 2: Structured output with Pydantic
class ProductReview(BaseModel):
    """Structured product review model"""
    product_name: str = Field(description="Name of the product")
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")
    recommend: bool = Field(description="Would you recommend this product?")


async def structured_output_example():
    """Extract structured data from text using Pydantic models"""
    print("\n=== Structured Output Example ===")
    
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    review_text = """
    I recently bought the UltraPhone X and have been using it for a month.
    The camera is absolutely amazing, especially in low light. The battery 
    life is incredible - easily lasts 2 days. The screen is bright and clear.
    However, it's quite expensive and the phone is a bit heavy. Also, the 
    fingerprint sensor is sometimes slow. Overall, I'm happy with my purchase
    and would recommend it to others who can afford it.
    """
    
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="Extract structured product review information from the user's text."
        ),
        Message(role=MessageRole.USER, content=review_text)
    ]
    
    try:
        review = await client.complete(
            messages=messages,
            response_model=ProductReview,
            temperature=0.3
        )
        
        print(f"Product: {review.product_name}")
        print(f"Rating: {'⭐' * review.rating}")
        print(f"Pros: {', '.join(review.pros)}")
        print(f"Cons: {', '.join(review.cons)}")
        print(f"Summary: {review.summary}")
        print(f"Recommended: {'Yes' if review.recommend else 'No'}")
    except Exception as e:
        print(f"Error: {e}")


# Example 3: Streaming responses
async def streaming_example():
    """Stream responses token by token"""
    print("\n=== Streaming Example ===")
    
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a short story about a robot learning to paint (3 sentences)."
        )
    ]
    
    print("Streaming response: ", end="", flush=True)
    
    try:
        full_response = ""
        async for chunk in client.stream(
            messages=messages,
            temperature=0.8,
            max_tokens=200
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        print(f"\n\nTotal length: {len(full_response)} characters")
    except Exception as e:
        print(f"\nStreaming error: {e}")


# Example 4: Using middleware (retry, rate limiting)
async def middleware_example():
    """Demonstrate middleware features like retry and rate limiting"""
    print("\n=== Middleware Example ===")
    
    # Configure with custom middleware settings
    config = LLMConfig(
        retry_max_attempts=3,
        retry_initial_delay=1.0,
        rate_limit_requests_per_minute=10,
        enable_logging=True,
        enable_metrics=True
    )
    
    manager = LLMManager(config=config)
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # This will automatically retry on failure
    messages = [
        Message(role=MessageRole.USER, content="What is quantum computing?")
    ]
    
    try:
        response = await client.complete(
            messages=messages,
            max_tokens=150
        )
        print(f"Response with middleware: {response.content[:200]}...")
        
        # Get metrics
        metrics = manager.get_metrics()
        print(f"\nMetrics:")
        print(f"  Total requests: {metrics.get('total_requests', 0)}")
        print(f"  Total tokens: {metrics.get('total_tokens', 0)}")
        print(f"  Average latency: {metrics.get('average_latency_ms', 0):.2f}ms")
    except Exception as e:
        print(f"Error: {e}")


# Example 5: Connection pooling for concurrent requests
async def connection_pool_example():
    """Demonstrate connection pooling with concurrent requests"""
    print("\n=== Connection Pool Example ===")
    
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # Create multiple concurrent requests
    questions = [
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
        "When was the internet invented?",
        "What is DNA?"
    ]
    
    async def ask_question(question: str) -> str:
        messages = [Message(role=MessageRole.USER, content=question)]
        try:
            response = await client.complete(messages=messages, max_tokens=50)
            return f"{question} -> {response.content[:50]}..."
        except Exception as e:
            return f"{question} -> Error: {e}"
    
    # Run all questions concurrently
    start_time = datetime.now()
    results = await asyncio.gather(*[ask_question(q) for q in questions])
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("Concurrent responses:")
    for result in results:
        print(f"  {result}")
    print(f"\nCompleted {len(questions)} requests in {elapsed:.2f} seconds")


# Example 6: Multi-turn conversation with context
async def conversation_example():
    """Maintain context across multiple turns"""
    print("\n=== Multi-turn Conversation Example ===")
    
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # Build conversation history
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful math tutor. Be concise."
        )
    ]
    
    # Turn 1
    messages.append(Message(role=MessageRole.USER, content="What is calculus?"))
    response = await client.complete(messages=messages, max_tokens=100)
    print(f"User: What is calculus?")
    print(f"Assistant: {response.content}\n")
    messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))
    
    # Turn 2
    messages.append(Message(role=MessageRole.USER, content="What is a derivative?"))
    response = await client.complete(messages=messages, max_tokens=100)
    print(f"User: What is a derivative?")
    print(f"Assistant: {response.content}\n")
    messages.append(Message(role=MessageRole.ASSISTANT, content=response.content))
    
    # Turn 3
    messages.append(Message(role=MessageRole.USER, content="Can you give an example?"))
    response = await client.complete(messages=messages, max_tokens=100)
    print(f"User: Can you give an example?")
    print(f"Assistant: {response.content}")
    
    print(f"\nTotal messages in conversation: {len(messages)}")


# Example 7: Custom token management
class CodeGeneration(BaseModel):
    """Model for code generation"""
    language: str = Field(description="Programming language")
    code: str = Field(description="Generated code")
    explanation: str = Field(description="Explanation of the code")


async def token_management_example():
    """Demonstrate token counting and management"""
    print("\n=== Token Management Example ===")
    
    manager = LLMManager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a Python function to calculate factorial"
        )
    ]
    
    # Estimate tokens before making request
    from src.llm.middleware.token_calculator import TokenCalculator
    calculator = TokenCalculator()
    
    estimated_input = calculator.count_tokens(
        [m.content for m in messages],
        LLMProvider.ANTHROPIC
    )
    print(f"Estimated input tokens: {estimated_input}")
    
    try:
        response = await client.complete(
            messages=messages,
            response_model=CodeGeneration,
            max_tokens=300
        )
        
        print(f"\nGenerated {response.language} code:")
        print(response.code)
        print(f"\nExplanation: {response.explanation}")
        
        if response.usage:
            print(f"\nActual token usage:")
            print(f"  Input: {response.usage.input_tokens}")
            print(f"  Output: {response.usage.output_tokens}")
            print(f"  Total: {response.usage.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("LLM MODULE USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    await basic_completion_example()
    await structured_output_example()
    await streaming_example()
    await middleware_example()
    await connection_pool_example()
    await conversation_example()
    await token_management_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())