#!/usr/bin/env python3
"""
Simple interactive chat with GenAI provider.
This bypasses the complex copilot workflow and just does direct chat.
"""

import asyncio
import logging
import os
from pathlib import Path
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
    level=logging.WARNING,  # Reduce verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main function for simple interactive chat."""
    
    print("=" * 60)
    print("SIMPLE GENAI CHAT")
    print("=" * 60)
    print()
    print("A simple interactive chat using the GenAI provider")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("-" * 60)
    print()
    
    try:
        # Initialize the LLM manager
        manager = await get_llm_manager()
        
        # Get the default client (should be GenAI)
        client = manager.get_client()
        print(f"✓ Connected to {client.pool.config.provider.value}")
        print(f"✓ Model: {client.pool.config.model}")
        print()
        print("You can now start chatting!")
        print("-" * 60)
        print()
        
        # Maintain conversation history
        messages: List[Message] = []
        
        # Add a system message for context
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful AI assistant. Be concise and friendly."
        ))
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession ended.")
                break
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                # Keep only the system message
                messages = [messages[0]]
                print("\n✓ Conversation history cleared.\n")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Add user message
            messages.append(Message(
                role=MessageRole.USER,
                content=user_input
            ))
            
            try:
                # Get AI response
                response = await client.complete(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                
                # Display response
                print(f"\nAI: {response.content}")
                
                # Add to history
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content
                ))
                
                # Show token usage in subtle way
                if response.usage:
                    total = response.usage.get('total_tokens', 0)
                    print(f"[tokens: {total}]")
                
                print()
                
            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Please try again.\n")
                # Remove the failed user message
                messages.pop()
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())