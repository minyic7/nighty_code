#!/usr/bin/env python3
"""
Interactive test script for the Copilot module using GenAI provider.
This script creates an interactive session where you can chat with the AI.
"""

import asyncio
import logging
import os
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Set the config file to use GenAI
os.environ['LLM_CONFIG_PATH'] = 'config/llm_genai.yaml'

from src.copilot import InteractiveCopilot
from src.copilot.core.types import CopilotConfig
from src.llm import get_llm_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some verbose logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

async def main():
    """Main function to run interactive copilot session."""
    
    print("=" * 60)
    print("INTERACTIVE COPILOT SESSION WITH GENAI")
    print("=" * 60)
    print()
    print("Configuration: config/llm_genai.yaml")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'clear' to clear the conversation history")
    print("-" * 60)
    print()
    
    try:
        # Initialize the LLM manager first
        print("Initializing LLM Manager...")
        manager = await get_llm_manager()
        
        # Check which providers are configured
        providers = manager.list_providers()
        print(f"Available providers: {[p.value for p in providers]}")
        
        # Get the default provider
        default_client = manager.get_client()
        print(f"Default provider: {default_client.pool.config.provider.value}")
        print()
        
        # Initialize the Copilot
        print("Initializing Copilot...")
        config = CopilotConfig(
            llm_provider="genai",
            verbose=True,
            enable_tools=True,
            enable_memory=True,
            enable_guardrails=True
        )
        copilot = InteractiveCopilot(config=config)
        
        # Initialize the copilot
        await copilot.initialize()
        
        print("✓ Copilot initialized successfully!")
        print()
        print("You can now start chatting. The AI will respond to your messages.")
        print("-" * 60)
        print()
        
        # Interactive loop
        conversation_history = []
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except KeyboardInterrupt:
                print("\n\nSession interrupted by user.")
                break
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nEnding session...")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\n✓ Conversation history cleared.\n")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            try:
                # Get response from copilot
                print("\nAI: ", end="", flush=True)
                
                # Process the message through copilot's workflow
                if copilot.workflow:
                    result = await copilot.workflow.process_message(user_input)
                    response = result.get("response", "I'm sorry, I couldn't process that message.")
                else:
                    response = "Workflow not initialized properly."
                
                print(response)
                
                # Add AI response to history
                conversation_history.append({"role": "assistant", "content": response})
                
                print()
                
            except Exception as e:
                print(f"\n✗ Error getting response: {e}")
                print(f"  Error type: {type(e).__name__}")
                print("\nNote: This might be due to network issues or API limits.")
                print()
    
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")
        print()
        print("Troubleshooting:")
        print("1. Check that config/llm_genai.yaml exists and is valid")
        print("2. Ensure you have network access to the GenAI endpoint")
        print("3. Verify your API key is correct")
        print("4. Check if you're on the corporate network (if required)")
        return
    
    print()
    print("=" * 60)
    print("SESSION ENDED")
    print("=" * 60)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())