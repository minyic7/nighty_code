#!/usr/bin/env python3
"""
Complex Questions Test for Copilot with OpenAI
Tests various types of complex queries including reasoning, math, coding, and analysis
"""

import asyncio
import time
from src.llm import get_llm_manager, LLMProvider, Message, MessageRole


async def test_reasoning_questions():
    """Test complex reasoning and logic questions"""
    print("=== Complex Reasoning Questions ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    questions = [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "A train leaves Station A at 60 mph heading east. Another train leaves Station B at 80 mph heading west. If the stations are 280 miles apart, when will they meet?",
        "What's the logical flaw in this statement: 'This statement is false'?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        try:
            start_time = time.time()
            response = await client.complete(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are a logical reasoning expert. Provide clear, step-by-step explanations."),
                    Message(role=MessageRole.USER, content=question)
                ],
                temperature=0.3,
                max_tokens=200
            )
            elapsed = time.time() - start_time
            
            print(f"A{i}: {response.content}")
            print(f"⏱️  Response time: {elapsed:.2f}s\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}\n")


async def test_coding_questions():
    """Test programming and algorithm questions"""
    print("\n=== Complex Coding Questions ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    questions = [
        "Write a Python function to find the longest palindromic substring in a given string. Include time complexity.",
        "Explain the difference between deep copy and shallow copy in Python with examples.",
        "How would you implement a LRU (Least Recently Used) cache in Python? Describe the data structures needed.",
        "What's the time complexity of binary search and why? When would you NOT use binary search?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        try:
            response = await client.complete(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are a Python expert. Provide concise but complete answers with code examples when relevant."),
                    Message(role=MessageRole.USER, content=question)
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            print(f"A{i}: {response.content}\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}\n")


async def test_analytical_questions():
    """Test analytical and multi-step problem solving"""
    print("\n=== Complex Analytical Questions ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    questions = [
        "A company's revenue grew by 20% in year 1, fell by 10% in year 2, and grew by 15% in year 3. What's the total percentage change over the 3 years?",
        "Analyze the pros and cons of using microservices vs monolithic architecture for a startup with 5 developers.",
        "If a bacterial culture doubles every 3 hours and starts with 100 bacteria, how many bacteria will there be after 24 hours? Show the calculation.",
        "Compare the space and time complexity of quicksort, mergesort, and heapsort. Which would you choose for sorting 1 billion integers and why?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        try:
            response = await client.complete(
                messages=[
                    Message(role=MessageRole.USER, content=question)
                ],
                temperature=0.4,
                max_tokens=250
            )
            
            print(f"A{i}: {response.content}\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}\n")


async def test_creative_questions():
    """Test creative and open-ended questions"""
    print("\n=== Creative & Design Questions ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    questions = [
        "Design a simple URL shortener system. What are the main components and potential scalability issues?",
        "How would you explain recursion to a 10-year-old using a real-world analogy?",
        "Propose 3 innovative features for a code editor specifically designed for pair programming.",
        "Create a haiku about debugging code, then explain what makes it a proper haiku."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        
        try:
            response = await client.complete(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="Be creative and thorough in your responses."),
                    Message(role=MessageRole.USER, content=question)
                ],
                temperature=0.7,  # Higher temperature for creativity
                max_tokens=250
            )
            
            print(f"A{i}: {response.content}\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error: {e}\n")


async def test_conversation_memory():
    """Test multi-turn conversation with context retention"""
    print("\n=== Multi-turn Conversation Test ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    # Build a conversation about a specific topic
    conversation = [
        Message(role=MessageRole.SYSTEM, content="You are helping design a distributed system. Remember previous context in the conversation.")
    ]
    
    user_messages = [
        "I want to build a distributed task queue system. What are the key components I need?",
        "For the component you mentioned first, what technology would you recommend?",
        "How would I handle failures in that component?",
        "Can you summarize the system we've discussed in 2-3 sentences?"
    ]
    
    for i, user_msg in enumerate(user_messages, 1):
        print(f"User {i}: {user_msg}")
        
        conversation.append(Message(role=MessageRole.USER, content=user_msg))
        
        try:
            response = await client.complete(
                messages=conversation,
                temperature=0.4,
                max_tokens=200
            )
            
            print(f"Assistant {i}: {response.content}\n")
            
            # Add assistant response to conversation history
            conversation.append(Message(role=MessageRole.ASSISTANT, content=response.content))
            
        except Exception as e:
            print(f"Error: {e}\n")
            break


async def test_performance_metrics():
    """Test response times and consistency"""
    print("\n=== Performance Metrics Test ===\n")
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.OPENAI)
    
    # Same question asked multiple times to test consistency
    test_question = "What are the three most important principles of clean code?"
    
    response_times = []
    responses = []
    
    print(f"Testing consistency with question: {test_question}\n")
    print("Running 3 iterations...\n")
    
    for i in range(3):
        try:
            start_time = time.time()
            response = await client.complete(
                messages=[
                    Message(role=MessageRole.USER, content=test_question)
                ],
                temperature=0.2,  # Low temperature for consistency
                max_tokens=150
            )
            elapsed = time.time() - start_time
            
            response_times.append(elapsed)
            responses.append(response.content)
            
            print(f"Iteration {i+1}:")
            print(f"  Response: {response.content[:100]}...")
            print(f"  Time: {elapsed:.2f}s\n")
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}\n")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"📊 Performance Summary:")
        print(f"  Average response time: {avg_time:.2f}s")
        print(f"  Fastest: {min(response_times):.2f}s")
        print(f"  Slowest: {max(response_times):.2f}s")
        
        # Check consistency (simplified - just checking if responses start similarly)
        if len(responses) > 1:
            first_words = [r.split()[:5] for r in responses]
            if all(fw == first_words[0] for fw in first_words):
                print("  ✅ Responses are highly consistent")
            else:
                print("  ⚠️  Some variation in responses (expected with creative tasks)")


async def main():
    print("=" * 70)
    print("COMPLEX QUESTIONS TEST - OpenAI GPT-3.5-turbo")
    print("=" * 70 + "\n")
    
    # Track total time
    total_start = time.time()
    
    try:
        # Run all test categories
        await test_reasoning_questions()
        await test_coding_questions()
        await test_analytical_questions()
        await test_creative_questions()
        await test_conversation_memory()
        await test_performance_metrics()
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print(f"Total test time: {total_elapsed:.2f} seconds")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())