#!/usr/bin/env python3
"""
Test GenAI provider with Instructor for structured output.
This tests that the GenAI provider works correctly with Instructor.
"""

import asyncio
import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Set the config file to use GenAI
os.environ['LLM_CONFIG_PATH'] = 'config/llm_genai.yaml'

from src.llm import get_llm_manager
from src.llm.core.types import Message, MessageRole

# Set up logging  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some verbose logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Define structured output models
class MathAnswer(BaseModel):
    """A mathematical answer with reasoning"""
    question: str = Field(description="The original question")
    reasoning: str = Field(description="Step-by-step reasoning")
    answer: float = Field(description="The numerical answer")

class JokeResponse(BaseModel):
    """A joke with explanation"""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")
    explanation: str = Field(description="Why it's funny")

class TodoItem(BaseModel):
    """A single todo item"""
    task: str = Field(description="What needs to be done")
    priority: str = Field(description="Priority level: high, medium, or low")
    estimated_minutes: int = Field(description="Estimated time in minutes")

class TodoList(BaseModel):
    """A list of todos"""
    items: List[TodoItem] = Field(description="List of todo items")
    total_time_minutes: int = Field(description="Total estimated time")

async def main():
    """Main function to test GenAI with Instructor."""
    
    print("=" * 60)
    print("GENAI PROVIDER WITH INSTRUCTOR TEST")
    print("=" * 60)
    print()
    print("Configuration: config/llm_genai.yaml")
    print("Testing structured output with Pydantic models")
    print("-" * 60)
    print()
    
    try:
        # Initialize the LLM manager
        print("Initializing LLM Manager...")
        manager = await get_llm_manager()
        
        # Get the default client (should be GenAI)
        client = manager.get_client()
        print(f"✓ Provider: {client.pool.config.provider.value}")
        print(f"✓ Model: {client.pool.config.model}")
        print()
        
        # Test 1: Math problem with structured output
        print("Test 1: Math Problem")
        print("-" * 40)
        messages = [
            Message(
                role=MessageRole.USER,
                content="What is 15 + 27? Explain your reasoning."
            )
        ]
        
        try:
            result = await client.complete(
                messages=messages,
                response_model=MathAnswer,
                temperature=0.3
            )
            print(f"Question: {result.question}")
            print(f"Reasoning: {result.reasoning}")
            print(f"Answer: {result.answer}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 2: Joke generation
        print("Test 2: Joke Generation")
        print("-" * 40)
        messages = [
            Message(
                role=MessageRole.USER,
                content="Tell me a joke about programming"
            )
        ]
        
        try:
            result = await client.complete(
                messages=messages,
                response_model=JokeResponse,
                temperature=0.8
            )
            print(f"Setup: {result.setup}")
            print(f"Punchline: {result.punchline}")
            print(f"Why it's funny: {result.explanation}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 3: Todo list generation
        print("Test 3: Todo List Generation")
        print("-" * 40)
        messages = [
            Message(
                role=MessageRole.USER,
                content="Create a todo list for setting up a new Python project"
            )
        ]
        
        try:
            result = await client.complete(
                messages=messages,
                response_model=TodoList,
                temperature=0.5
            )
            print(f"Total time: {result.total_time_minutes} minutes")
            print("Tasks:")
            for i, item in enumerate(result.items, 1):
                print(f"  {i}. [{item.priority}] {item.task} ({item.estimated_minutes} min)")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        print("=" * 60)
        print("INSTRUCTOR TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())