#!/usr/bin/env python3
"""
Test script specifically for testing base URL functionality
This simulates what happens with the GenAI endpoint
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_direct_openai_with_base_url():
    """Test OpenAI client directly with custom base URL"""
    print("\n" + "="*60)
    print("DIRECT OPENAI CLIENT TEST WITH BASE URL")
    print("="*60)
    
    try:
        from openai import AsyncOpenAI
        
        # Create client with custom base URL
        client = AsyncOpenAI(
            base_url="https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
            api_key="sk-EiK-aYVNF0o1WoTJOnGyCg"
        )
        
        print(f"✓ Created AsyncOpenAI client")
        print(f"  - Base URL: {client.base_url}")
        print(f"  - API Key: sk-***{client.api_key[-5:]}")
        
        # Test a simple completion
        try:
            response = await client.chat.completions.create(
                model="guardrails-bedrock-claude-4-sonnet",
                messages=[{"role": "user", "content": "Say hello!"}],
                max_tokens=50
            )
            
            print(f"✓ Completion successful!")
            print(f"  Response: {response.choices[0].message.content}")
            print(f"  Model: {response.model}")
            if hasattr(response, 'usage'):
                print(f"  Usage: {response.usage}")
                
        except Exception as e:
            print(f"✗ Completion failed (expected without network): {e}")
            print(f"  Error type: {type(e).__name__}")
            
        finally:
            await client.close()
            print(f"✓ Client closed")
            
    except Exception as e:
        print(f"✗ Direct client test error: {e}")
        import traceback
        traceback.print_exc()


async def test_llm_module_with_base_url():
    """Test LLM module with custom base URL"""
    print("\n" + "="*60)
    print("LLM MODULE TEST WITH BASE URL")
    print("="*60)
    
    try:
        from src.llm import (
            LLMProvider,
            LLMConfig,
            Message,
            MessageRole,
            CompletionRequest,
            LLMManager,
            PoolConfig
        )
        
        # Create manager
        manager = LLMManager()
        
        # Create GenAI config with base URL
        genai_config = LLMConfig(
            provider=LLMProvider.GENAI,
            api_key="sk-EiK-aYVNF0o1WoTJOnGyCg",
            model="guardrails-bedrock-claude-4-sonnet",
            base_url="https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
            temperature=0.7,
            max_tokens=100,
            timeout=60,
            max_retries=3
        )
        
        print(f"✓ Created GenAI config")
        print(f"  - Provider: {genai_config.provider.value}")
        print(f"  - Model: {genai_config.model}")
        print(f"  - Base URL: {genai_config.base_url}")
        
        # Initialize manager
        await manager.initialize()
        print(f"✓ Manager initialized")
        
        # Add provider with custom pool config
        pool_config = PoolConfig(
            min_size=1,
            max_size=3,
            acquire_timeout=30.0,
            idle_timeout=3600.0
        )
        
        await manager.add_provider(genai_config, pool_config)
        print(f"✓ Added GenAI provider with custom pool config")
        
        # Get client
        client = manager.get_client(LLMProvider.GENAI)
        print(f"✓ Retrieved client")
        
        # Check pool details
        pool = client.pool
        print(f"\n✓ Pool Details:")
        print(f"  - Pool ID: {id(pool)}")
        print(f"  - Min size: {pool._min_size}")
        print(f"  - Max size: {pool._max_size}")
        print(f"  - Configs count: {len(pool._configs)}")
        
        # Check provider details
        if pool._clients:
            provider = list(pool._clients.values())[0]
            print(f"\n✓ Provider Details:")
            print(f"  - Provider type: {type(provider).__name__}")
            print(f"  - Client ID: {provider._client_id}")
            print(f"  - Status: {provider._status}")
            
            # Check the underlying OpenAI client
            if hasattr(provider, 'client'):
                print(f"\n✓ Underlying Client Details:")
                print(f"  - Client type: {type(provider.client).__name__}")
                if hasattr(provider.client, 'base_url'):
                    print(f"  - Base URL configured: {provider.client.base_url}")
        
        # Test completion
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Hello, GenAI!")
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        print(f"\n✓ Created completion request")
        
        try:
            response = await client.complete(request)
            print(f"✓ Completion successful!")
            print(f"  Response: {response.content}")
            print(f"  Provider: {response.provider.value}")
            print(f"  Model: {response.model}")
        except Exception as e:
            print(f"✗ Completion failed (expected without network): {e}")
            
        # Clean up
        await manager.close()
        print(f"\n✓ Manager closed")
        
    except Exception as e:
        print(f"✗ LLM module test error: {e}")
        import traceback
        traceback.print_exc()


async def test_provider_initialization_flow():
    """Debug the exact initialization flow for GenAI provider"""
    print("\n" + "="*60)
    print("PROVIDER INITIALIZATION FLOW DEBUG")
    print("="*60)
    
    try:
        from src.llm.providers.genai import GenAIProvider
        from src.llm import LLMConfig, LLMProvider
        
        # Create config
        config = LLMConfig(
            provider=LLMProvider.GENAI,
            api_key="sk-EiK-aYVNF0o1WoTJOnGyCg",
            model="guardrails-bedrock-claude-4-sonnet",
            base_url="https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
            temperature=0.7,
            timeout=60
        )
        
        print(f"✓ Config created:")
        print(f"  - Provider: {config.provider}")
        print(f"  - Model: {config.model}")
        print(f"  - Base URL: {config.base_url}")
        
        # Create provider instance
        provider = GenAIProvider(config)
        print(f"\n✓ Provider instance created:")
        print(f"  - Type: {type(provider).__name__}")
        print(f"  - Client ID: {provider._client_id}")
        print(f"  - Initial status: {provider._status.is_available}")
        
        # Initialize provider
        await provider.initialize()
        print(f"\n✓ Provider initialized:")
        print(f"  - Status after init: {provider._status.is_available}")
        
        # Check the client
        if hasattr(provider, 'client'):
            print(f"\n✓ OpenAI client details:")
            print(f"  - Client exists: {provider.client is not None}")
            print(f"  - Client type: {type(provider.client).__name__}")
            if hasattr(provider.client, 'base_url'):
                print(f"  - Base URL: {provider.client.base_url}")
            if hasattr(provider.client, 'api_key'):
                print(f"  - API Key: sk-***{provider.client.api_key[-5:]}")
        
        # Close provider
        await provider.close()
        print(f"\n✓ Provider closed")
        
    except Exception as e:
        print(f"✗ Initialization flow error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all base URL tests"""
    print("\n" + "="*60)
    print("BASE URL TESTING SUITE")
    print("="*60)
    print("Testing custom base URL functionality for GenAI integration")
    
    await test_direct_openai_with_base_url()
    await test_provider_initialization_flow()
    await test_llm_module_with_base_url()
    
    print("\n" + "="*60)
    print("BASE URL TESTS COMPLETE")
    print("="*60)
    print("\nTo run on company laptop:")
    print("1. Ensure you're connected to the corporate network")
    print("2. The GenAI endpoint should be accessible")
    print("3. If using proxy, ensure it's configured correctly")
    print("4. Check if you need any additional certificates")


if __name__ == "__main__":
    asyncio.run(main())