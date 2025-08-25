#!/usr/bin/env python3
"""
Simple Direct Q&A Test with OpenAI
"""

import asyncio
from src.llm import get_llm_manager, LLMProvider, Message, MessageRole


async def test_direct_openai():
    """Test OpenAI directly without copilot complexity"""
    print("=== Direct OpenAI GPT-3.5-turbo Test ===\n")
    
    # Get LLM manager
    manager = await get_llm_manager()
    
    # Get OpenAI client
    client = manager.get_client(LLMProvider.OPENAI)
    print("✅ OpenAI client obtained\n")
    
    # Simple questions
    questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Name three primary colors",
        "Complete this: Roses are red, violets are..."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        
        try:
            # Create messages list
            messages = [Message(role=MessageRole.USER, content=question)]
            
            # Call the LLM with messages directly
            response = await client.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=50
            )
            response_text = response.content if response else "No response"
            
            print(f"Answer: {response_text}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("✅ Test completed!")


async def test_with_system_message():
    """Test with system message for better control"""
    print("\n=== Test with System Message ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    # Questions with a system prompt to ensure brief answers
    system_message = Message(
        role=MessageRole.SYSTEM, 
        content="You are a helpful assistant. Give very brief, direct answers."
    )
    
    questions = [
        "What programming language has a snake logo?",
        "What is HTTP status code 404?",
        "What does CPU stand for?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        
        try:
            messages = [
                system_message,
                Message(role=MessageRole.USER, content=question)
            ]
            
            response_obj = await client.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=30
            )
            response = response_obj.content if response_obj else "No response"
            
            print(f"A: {response}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("✅ System message test completed!")


async def main():
    print("=" * 60)
    print("SIMPLE OPENAI Q&A TEST")
    print("Using: GPT-3.5-turbo")
    print("=" * 60 + "\n")
    
    try:
        await test_direct_openai()
        await test_with_system_message()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())