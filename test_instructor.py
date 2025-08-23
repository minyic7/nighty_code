#!/usr/bin/env python3
"""Test script for LLM module with Instructor integration"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import get_llm_manager, LLMProvider, Message, MessageRole


# Define Pydantic models for structured outputs
class MathProblem(BaseModel):
    """Structured math problem analysis"""
    problem: str = Field(description="The original problem statement")
    steps: List[str] = Field(description="Step-by-step solution")
    answer: float = Field(description="The final numerical answer")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer (0-1)")


class CodeAnalysis(BaseModel):
    """Structured code analysis"""
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    functions: List[str] = Field(description="List of function names")
    has_errors: bool = Field(description="Whether the code has errors")
    suggestions: Optional[List[str]] = Field(description="Improvement suggestions")


class TaskPlan(BaseModel):
    """Structured task planning"""
    user_intent: str = Field(description="What the user wants to achieve")
    requires_tools: bool = Field(description="Whether external tools are needed")
    steps: List[str] = Field(description="Ordered list of steps to complete")
    estimated_complexity: str = Field(description="simple, medium, or complex")
    potential_issues: List[str] = Field(description="Potential problems to watch for")


async def test_basic_completion():
    """Test basic completion without Instructor"""
    print("\n" + "="*60)
    print("TEST 1: Basic Completion (No Instructor)")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(MessageRole.USER, "What is 2 + 2?")
    ]
    
    # Normal completion - returns CompletionResponse
    response = await client.complete(messages, max_tokens=100)
    print(f"Type returned: {type(response).__name__}")
    print(f"Response: {response.content}")


async def test_structured_math():
    """Test structured output with math problem"""
    print("\n" + "="*60)
    print("TEST 2: Structured Math Problem (With Instructor)")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(MessageRole.USER, 
                "Solve this problem: A train travels 120 miles in 2 hours. "
                "Then it speeds up and travels 180 miles in 2.5 hours. "
                "What is the average speed for the entire journey?")
    ]
    
    # Structured completion - returns MathProblem instance
    result = await client.complete(
        messages,
        response_model=MathProblem,  # This triggers Instructor!
        max_tokens=500
    )
    
    print(f"Type returned: {type(result).__name__}")
    print(f"\nProblem: {result.problem}")
    print(f"\nSteps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")
    print(f"\nAnswer: {result.answer} mph")
    print(f"Confidence: {result.confidence:.1%}")


async def test_code_analysis():
    """Test structured code analysis"""
    print("\n" + "="*60)
    print("TEST 3: Code Analysis (With Instructor)")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    code_snippet = '''
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    '''
    
    messages = [
        Message(MessageRole.USER, f"Analyze this code:\n```python\n{code_snippet}\n```")
    ]
    
    # Structured completion
    result = await client.complete(
        messages,
        response_model=CodeAnalysis,
        max_tokens=500
    )
    
    print(f"Language: {result.language}")
    print(f"Purpose: {result.purpose}")
    print(f"Functions: {', '.join(result.functions)}")
    print(f"Has Errors: {result.has_errors}")
    if result.suggestions:
        print("Suggestions:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion}")


async def test_task_planning():
    """Test task planning with structured output"""
    print("\n" + "="*60)
    print("TEST 4: Task Planning (With Instructor)")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    messages = [
        Message(MessageRole.USER,
                "I want to refactor a Python module to use async/await "
                "and add proper error handling")
    ]
    
    # Structured completion
    result = await client.complete(
        messages,
        response_model=TaskPlan,
        max_tokens=500
    )
    
    print(f"User Intent: {result.user_intent}")
    print(f"Requires Tools: {result.requires_tools}")
    print(f"Complexity: {result.estimated_complexity}")
    print("\nSteps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")
    print("\nPotential Issues:")
    for issue in result.potential_issues:
        print(f"  ⚠️ {issue}")


async def test_chat_with_structured():
    """Test the chat interface with structured output"""
    print("\n" + "="*60)
    print("TEST 5: Chat Interface with Structured Output")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # Using chat() with response_model
    result = await client.chat(
        "Explain how to make a sandwich",
        response_model=TaskPlan
    )
    
    print(f"Type returned: {type(result).__name__}")
    print(f"Steps to make a sandwich:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")


async def test_error_handling():
    """Test error handling when Instructor fails"""
    print("\n" + "="*60)
    print("TEST 6: Error Handling")
    print("="*60)
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    # Define a model with strict validation
    class StrictModel(BaseModel):
        exact_number: int = Field(ge=10, le=20, description="Must be between 10 and 20")
        
    messages = [
        Message(MessageRole.USER, "Give me a number")
    ]
    
    try:
        result = await client.complete(
            messages,
            response_model=StrictModel,
            max_tokens=100
        )
        print(f"Success! Number: {result.exact_number}")
    except Exception as e:
        print(f"Error (as expected): {type(e).__name__}: {e}")


async def main():
    """Run all tests"""
    print("\n🚀 Testing LLM Module with Instructor Integration")
    print("="*60)
    
    try:
        # Test without Instructor
        await test_basic_completion()
        
        # Test with Instructor
        await test_structured_math()
        await test_code_analysis()
        await test_task_planning()
        await test_chat_with_structured()
        await test_error_handling()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())