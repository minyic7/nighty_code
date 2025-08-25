#!/usr/bin/env python3
"""
Test Copilot with Different LLM Providers
"""

import asyncio
from src.copilot import InteractiveCopilot
from src.copilot.core.types import CopilotConfig
from src.llm import get_llm_manager, LLMProvider


async def test_provider_configuration():
    """Test that copilot respects the configured provider"""
    print("=== Testing Copilot Provider Configuration ===\n")
    
    # Check which providers are available
    manager = await get_llm_manager()
    available_providers = manager.list_providers()
    print(f"Available providers: {[p.value for p in available_providers]}\n")
    
    # Test with each available provider
    for provider in available_providers:
        print(f"Testing with {provider.value} provider:")
        
        try:
            # Create copilot with specific provider
            config = CopilotConfig(
                llm_provider=provider.value,
                enable_tools=False,  # Disable tools for simple test
                enable_memory=False,
                enable_guardrails=False
            )
            
            copilot = InteractiveCopilot(config=config)
            await copilot.initialize()
            
            print(f"✅ Successfully initialized copilot with {provider.value}")
            
            # Clean up
            await copilot.shutdown()
            
        except Exception as e:
            print(f"❌ Failed to initialize with {provider.value}: {e}")
        
        print("-" * 50)
    
    # Test with invalid provider (should fail gracefully)
    print("\nTesting with invalid provider:")
    try:
        config = CopilotConfig(
            llm_provider="invalid_provider",
            enable_tools=False
        )
        copilot = InteractiveCopilot(config=config)
        await copilot.initialize()
        print("❌ Should have failed with invalid provider")
    except Exception as e:
        print(f"✅ Correctly failed with invalid provider: {e}")


async def test_openai_copilot():
    """Test copilot specifically with OpenAI (if configured)"""
    print("\n=== Testing Copilot with OpenAI ===\n")
    
    manager = await get_llm_manager()
    
    # Check if OpenAI is configured
    if LLMProvider.OPENAI not in manager.list_providers():
        print("⚠️  OpenAI provider not configured")
        print("   Add OpenAI to your llm.yaml to test this provider")
        return
    
    try:
        # Create copilot with OpenAI
        config = CopilotConfig(
            llm_provider="openai",
            enable_tools=False,
            enable_memory=True
        )
        
        copilot = InteractiveCopilot(config=config)
        await copilot.initialize()
        
        print("✅ Copilot initialized with OpenAI provider")
        
        # Test a simple query
        if copilot.workflow:
            response = await copilot.workflow.process_message("Say 'Hello from OpenAI!'")
            if response:
                print(f"Response: {response.content[:100]}...")
        
        await copilot.shutdown()
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")


async def main():
    print("=" * 60)
    print("COPILOT PROVIDER TEST")
    print("=" * 60 + "\n")
    
    await test_provider_configuration()
    await test_openai_copilot()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())