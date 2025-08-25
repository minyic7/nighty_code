#!/usr/bin/env python3
"""
Comprehensive test script for LLM module with OpenAI and GenAI providers
This script tests the integration and provides detailed debug information
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging for detailed debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LLM module components
from src.llm import (
    LLMProvider,
    LLMConfig,
    Message,
    MessageRole,
    CompletionRequest,
    get_llm_manager,
    config_manager,
    ConfigManager,
    PoolConfig
)


async def test_openai_provider():
    """Test OpenAI provider with standard API"""
    print("\n" + "="*60)
    print("TESTING OPENAI PROVIDER")
    print("="*60)
    
    try:
        # Create OpenAI config
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="sk-test-key",  # Replace with actual key
            model="gpt-3.5-turbo",
            base_url=None,  # Use default OpenAI URL
            temperature=0.7,
            max_tokens=100,
            timeout=30,
            max_retries=3
        )
        
        print(f"✓ Created OpenAI config:")
        print(f"  - Provider: {openai_config.provider.value}")
        print(f"  - Model: {openai_config.model}")
        print(f"  - Base URL: {openai_config.base_url or 'default'}")
        print(f"  - Temperature: {openai_config.temperature}")
        print(f"  - Max tokens: {openai_config.max_tokens}")
        
        # Test with manager
        manager = await get_llm_manager()
        
        # Add the provider
        await manager.add_provider(openai_config)
        print(f"✓ Added OpenAI provider to manager")
        
        # Get client
        client = manager.get_client(LLMProvider.OPENAI)
        print(f"✓ Retrieved OpenAI client from manager")
        
        # Test completion
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Say hello in 5 words")
            ],
            temperature=0.7,
            max_tokens=20
        )
        
        print(f"✓ Created completion request")
        print(f"  Attempting to send request...")
        
        try:
            response = await client.complete(request)
            print(f"✓ OpenAI Response: {response.content}")
            print(f"  - Model used: {response.model}")
            print(f"  - Provider: {response.provider.value}")
            print(f"  - Tokens: {response.usage}")
        except Exception as e:
            print(f"✗ OpenAI completion failed (expected if no valid key): {e}")
            
    except Exception as e:
        print(f"✗ OpenAI test error: {e}")
        import traceback
        traceback.print_exc()


async def test_genai_provider():
    """Test GenAI provider with custom base URL"""
    print("\n" + "="*60)
    print("TESTING GENAI PROVIDER")
    print("="*60)
    
    try:
        # Create GenAI config
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
        
        print(f"✓ Created GenAI config:")
        print(f"  - Provider: {genai_config.provider.value}")
        print(f"  - Model: {genai_config.model}")
        print(f"  - Base URL: {genai_config.base_url}")
        print(f"  - Temperature: {genai_config.temperature}")
        print(f"  - Max tokens: {genai_config.max_tokens}")
        print(f"  - Timeout: {genai_config.timeout}s")
        
        # Test with manager
        manager = await get_llm_manager()
        
        # Add the provider
        await manager.add_provider(genai_config)
        print(f"✓ Added GenAI provider to manager")
        
        # Get client
        client = manager.get_client(LLMProvider.GENAI)
        print(f"✓ Retrieved GenAI client from manager")
        
        # Get pool information
        pool = client.pool
        print(f"✓ Connection pool info:")
        print(f"  - Min size: {pool._min_size}")
        print(f"  - Max size: {pool._max_size}")
        print(f"  - Current clients: {len(pool._clients)}")
        print(f"  - Available clients: {len(pool._available)}")
        
        # Test completion
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Say hello!")
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        print(f"✓ Created completion request for GenAI")
        print(f"  Attempting to send request...")
        
        try:
            response = await client.complete(request)
            print(f"✓ GenAI Response: {response.content}")
            print(f"  - Model used: {response.model}")
            print(f"  - Provider: {response.provider.value}")
            print(f"  - Client ID: {response.client_id}")
            print(f"  - Tokens: {response.usage}")
        except Exception as e:
            print(f"✗ GenAI completion failed (expected without network access): {e}")
            
    except Exception as e:
        print(f"✗ GenAI test error: {e}")
        import traceback
        traceback.print_exc()


async def test_config_loading():
    """Test loading configuration from YAML files"""
    print("\n" + "="*60)
    print("TESTING CONFIG LOADING")
    print("="*60)
    
    try:
        # Test loading standard config
        standard_config = ConfigManager(config_path="config/llm.yaml")
        print(f"✓ Loaded standard config")
        print(f"  - Default provider: {standard_config.global_config.default_provider}")
        print(f"  - Configured providers: {list(standard_config.global_config.providers.keys())}")
        
        # Test loading GenAI config
        genai_config = ConfigManager(config_path="config/llm_genai.yaml")
        print(f"✓ Loaded GenAI config")
        print(f"  - Default provider: {genai_config.global_config.default_provider}")
        print(f"  - Configured providers: {list(genai_config.global_config.providers.keys())}")
        
        # Check GenAI specific settings
        if LLMProvider.GENAI in genai_config.global_config.providers:
            genai_settings = genai_config.global_config.providers[LLMProvider.GENAI]
            print(f"✓ GenAI provider settings:")
            print(f"  - Model: {genai_settings.model}")
            print(f"  - Base URL: {genai_settings.base_url}")
            print(f"  - Timeout: {genai_settings.timeout}s")
            
    except Exception as e:
        print(f"✗ Config loading error: {e}")
        import traceback
        traceback.print_exc()


async def test_manager_debug_info():
    """Get detailed debug information about the LLM manager"""
    print("\n" + "="*60)
    print("LLM MANAGER DEBUG INFORMATION")
    print("="*60)
    
    try:
        manager = await get_llm_manager()
        
        print(f"✓ Manager initialized: {manager._initialized}")
        print(f"✓ Configured providers: {manager.list_providers()}")
        
        # Get status for each provider
        for provider in manager.list_providers():
            print(f"\n  Provider: {provider.value}")
            status = manager.get_status(provider)
            print(f"    - Pool status: {status}")
            
            # Get client info
            try:
                client = manager.get_client(provider)
                print(f"    - Client available: Yes")
                print(f"    - Pool size: {len(client.pool._clients)}")
                print(f"    - Available connections: {len(client.pool._available)}")
            except Exception as e:
                print(f"    - Client available: No ({e})")
                
    except Exception as e:
        print(f"✗ Manager debug error: {e}")
        import traceback
        traceback.print_exc()


async def test_provider_switching():
    """Test switching between providers"""
    print("\n" + "="*60)
    print("TESTING PROVIDER SWITCHING")
    print("="*60)
    
    try:
        # Initialize manager with both providers
        manager = await get_llm_manager()
        
        # Add OpenAI
        openai_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="sk-test-openai",
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        await manager.add_provider(openai_config)
        print(f"✓ Added OpenAI provider")
        
        # Add GenAI
        genai_config = LLMConfig(
            provider=LLMProvider.GENAI,
            api_key="sk-EiK-aYVNF0o1WoTJOnGyCg",
            model="guardrails-bedrock-claude-4-sonnet",
            base_url="https://genai-llm-gw.anigenailabs01.aws.prod.au.internal.cba",
            temperature=0.7
        )
        await manager.add_provider(genai_config)
        print(f"✓ Added GenAI provider")
        
        # Test getting different clients
        openai_client = manager.get_client(LLMProvider.OPENAI)
        print(f"✓ Got OpenAI client: {openai_client is not None}")
        
        genai_client = manager.get_client(LLMProvider.GENAI)
        print(f"✓ Got GenAI client: {genai_client is not None}")
        
        # Check they're different
        print(f"✓ Clients are different: {openai_client is not genai_client}")
        
        # Check pool configurations
        print(f"\nPool configurations:")
        print(f"  OpenAI pool ID: {id(openai_client.pool)}")
        print(f"  GenAI pool ID: {id(genai_client.pool)}")
        
    except Exception as e:
        print(f"✗ Provider switching error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("LLM MODULE COMPREHENSIVE TEST SUITE")
    print("="*60)
    print("This script tests the LLM module with OpenAI and GenAI providers")
    print("Note: Actual API calls will fail without network access to endpoints")
    print("The script focuses on configuration, initialization, and integration")
    
    # Run tests
    await test_config_loading()
    await test_openai_provider()
    await test_genai_provider()
    await test_provider_switching()
    await test_manager_debug_info()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("\nNotes for running on company laptop:")
    print("1. Ensure you're on the corporate network to access GenAI endpoint")
    print("2. Replace API keys with valid ones if needed")
    print("3. The GenAI base URL should be accessible from your network")
    print("4. Check firewall/proxy settings if connections fail")
    print("\nTo use GenAI as default, use config/llm_genai.yaml")
    print("To use OpenAI as default, use config/llm.yaml")


if __name__ == "__main__":
    asyncio.run(main())