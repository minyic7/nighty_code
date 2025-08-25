#!/usr/bin/env python3
"""
Direct test of GenAI provider without copilot complexity.
This tests basic LLM functionality with the GenAI provider.
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

async def main():
    """Main function to test GenAI provider directly."""
    
    print("=" * 60)
    print("DIRECT GENAI PROVIDER TEST")
    print("=" * 60)
    print()
    print("Configuration: config/llm_genai.yaml")
    print("-" * 60)
    print()
    
    try:
        # Initialize the LLM manager
        print("Initializing LLM Manager...")
        manager = await get_llm_manager()
        
        # Check which providers are configured
        providers = manager.list_providers()
        print(f"✓ Available providers: {[p.value for p in providers]}")
        
        # Get the default client (should be GenAI)
        client = manager.get_client()
        print(f"✓ Default provider: {client.pool.config.provider.value}")
        print(f"✓ Model: {client.pool.config.model}")
        print(f"✓ Base URL: {client.pool.config.base_url}")
        print()
        
        # Test questions
        test_questions = [
            "What is 2 + 2? Answer with just the number.",
            "What is the capital of France? Answer in one word.",
            "Tell me a very short joke (one sentence)."
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"Test {i}: {question}")
            print("-" * 40)
            
            # Prepare messages
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=question
                )
            ]
            
            try:
                # Make the completion request
                response = await client.complete(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=50
                )
                
                print(f"Response: {response.content}")
                
                # Show token usage if available
                if response.usage:
                    print(f"Tokens: {response.usage.get('total_tokens', 'N/A')}")
                
                # Show model info
                if response.model:
                    print(f"Model: {response.model}")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                print(f"  Type: {type(e).__name__}")
            
            print()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print(f"  Error type: {type(e).__name__}")
        print()
        print("Note: This error is expected if you're not on the corporate network.")
        print("The GenAI endpoint may require corporate network access.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())