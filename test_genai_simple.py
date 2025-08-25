#!/usr/bin/env python3
"""
Simple test script to ask a question using the LLM module with GenAI provider.
"""

import asyncio
import logging
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.llm import get_llm_manager, LLMConfig, LLMProvider
from src.llm.core.types import LLMMessage, MessageRole

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main function to test GenAI provider with a simple question."""
    
    print("=" * 60)
    print("SIMPLE GENAI LLM TEST")
    print("=" * 60)
    print()
    
    try:
        # Create GenAI configuration
        genai_config = LLMConfig(
            provider=LLMProvider.GENAI,
            model="guardrails-bedrock-claude-4-sonnet",
            base_url="https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
            api_key="sk-proj-abcddefgh1234567890",  # Replace with actual key if needed
            temperature=0.7,
            max_tokens=150,
            timeout=60
        )
        
        print("✓ Created GenAI configuration")
        print(f"  - Provider: {genai_config.provider.value}")
        print(f"  - Model: {genai_config.model}")
        print(f"  - Base URL: {genai_config.base_url}")
        print(f"  - Temperature: {genai_config.temperature}")
        print(f"  - Max tokens: {genai_config.max_tokens}")
        print()
        
        # Initialize the LLM manager
        manager = await get_llm_manager()
        print("✓ LLM Manager initialized")
        
        # Add the GenAI provider
        await manager.add_provider(genai_config)
        print("✓ Added GenAI provider to manager")
        print()
        
        # Get the GenAI client
        client = manager.get_client(LLMProvider.GENAI)
        print("✓ Retrieved GenAI client")
        print()
        
        # Prepare a simple question
        messages = [
            LLMMessage(
                role=MessageRole.USER,
                content="What is the capital of France? Please answer in one sentence."
            )
        ]
        
        print("Sending question to GenAI:")
        print(f"  Question: {messages[0].content}")
        print()
        
        # Make the completion request
        print("Attempting to get response...")
        response = await client.complete(
            messages=messages,
            temperature=0.7,
            max_tokens=50
        )
        
        # Display the response
        print("✓ Received response from GenAI!")
        print()
        print("Response:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        print()
        
        # Show token usage if available
        if response.usage:
            print("Token Usage:")
            print(f"  - Prompt tokens: {response.usage.get('prompt_tokens', 'N/A')}")
            print(f"  - Completion tokens: {response.usage.get('completion_tokens', 'N/A')}")
            print(f"  - Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        
        # Show model info
        if response.model:
            print(f"  - Model used: {response.model}")
        if response.provider:
            print(f"  - Provider: {response.provider}")
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print(f"  Error type: {type(e).__name__}")
        print()
        print("Note: This error is expected if you're not on the corporate network.")
        print("The GenAI endpoint requires corporate network access.")
        return
    
    print()
    print("=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())