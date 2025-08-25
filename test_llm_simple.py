#!/usr/bin/env python3
"""
Simple LLM Test - One basic query
"""

import asyncio
from src.llm import get_llm_manager, LLMProvider, Message, MessageRole


async def simple_test():
    """Test one simple query with the configured LLM"""
    
    print("=" * 50)
    print("SIMPLE LLM TEST")
    print("=" * 50 + "\n")
    
    # Get the LLM manager
    manager = await get_llm_manager()
    
    # Check what's configured
    providers = manager.list_providers()
    print(f"📋 Configured providers: {[p.value for p in providers]}")
    
    # Get the default client (will use OpenAI based on config)
    client = manager.get_client()
    print(f"✅ Using default provider from config\n")
    
    # Simple test query
    question = "What is Python in one sentence?"
    print(f"❓ Question: {question}")
    
    try:
        # Make the LLM call
        messages = [
            Message(role=MessageRole.USER, content=question)
        ]
        
        response = await client.complete(
            messages=messages,
            temperature=0.5,
            max_tokens=50
        )
        
        print(f"💬 Answer: {response.content}")
        print(f"\n✅ Test successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    return True


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(simple_test())
    exit(0 if success else 1)