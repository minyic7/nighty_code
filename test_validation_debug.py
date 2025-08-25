#!/usr/bin/env python3
"""
Debug OpenAI Client Validation Issues
This script tests the validation process step by step
"""

import asyncio
import os
from src.llm import get_llm_manager, LLMProvider


async def test_validation():
    """Test OpenAI validation in detail"""
    
    print("=" * 80)
    print("OPENAI VALIDATION DEBUG TEST")
    print("=" * 80 + "\n")
    
    # First, test if we can import and use OpenAI directly
    print("1️⃣  Testing Direct OpenAI Connection\n")
    
    try:
        from openai import AsyncOpenAI
        print("✅ AsyncOpenAI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AsyncOpenAI: {e}")
        return
    
    # Get the configuration
    print("\n2️⃣  Getting Configuration\n")
    
    manager = await get_llm_manager()
    config_mgr = manager._config_manager
    
    # Find OpenAI config
    openai_config = None
    for cfg in config_mgr.provider_configs:
        if cfg.provider == LLMProvider.OPENAI:
            openai_config = cfg
            break
    
    if not openai_config:
        print("❌ No OpenAI configuration found")
        return
    
    print(f"✅ Found OpenAI config:")
    print(f"   Model: {openai_config.model}")
    print(f"   Base URL: {openai_config.base_url or 'Default (https://api.openai.com/v1)'}")
    print(f"   API Key: ***...{openai_config.api_key[-10:] if openai_config.api_key else 'None'}")
    print(f"   Timeout: {openai_config.timeout}s")
    
    # Test creating the client
    print("\n3️⃣  Creating AsyncOpenAI Client\n")
    
    try:
        client = AsyncOpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url,
            timeout=openai_config.timeout,
            max_retries=openai_config.max_retries,
        )
        print("✅ AsyncOpenAI client created")
        print(f"   Client type: {type(client).__name__}")
        print(f"   Base URL: {client.base_url if hasattr(client, 'base_url') else 'Not accessible'}")
    except Exception as e:
        print(f"❌ Failed to create AsyncOpenAI client: {e}")
        return
    
    # Test the validation call that's failing
    print("\n4️⃣  Testing Validation Call (models.retrieve)\n")
    
    print(f"Attempting to retrieve model: {openai_config.model}")
    
    try:
        response = await client.models.retrieve(openai_config.model)
        print(f"✅ Model retrieved successfully!")
        print(f"   Model ID: {response.id}")
        print(f"   Model matches config: {response.id == openai_config.model}")
    except Exception as e:
        print(f"❌ Validation call failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Check for specific error patterns
        error_str = str(e).lower()
        if "connection" in error_str or "network" in error_str:
            print("\n⚠️  NETWORK/CONNECTION ISSUE DETECTED")
            print("   Possible causes:")
            print("   - Corporate firewall blocking api.openai.com")
            print("   - Proxy not configured for AsyncOpenAI")
            print("   - SSL/TLS certificate issues")
        elif "401" in str(e) or "authentication" in error_str or "api key" in error_str:
            print("\n⚠️  AUTHENTICATION ISSUE DETECTED")
            print("   Possible causes:")
            print("   - Invalid API key")
            print("   - API key not authorized for this model")
            print("   - API key expired or revoked")
        elif "404" in str(e) or "not found" in error_str:
            print("\n⚠️  MODEL NOT FOUND")
            print("   Possible causes:")
            print(f"   - Model '{openai_config.model}' doesn't exist")
            print("   - Wrong base URL (using different API endpoint)")
        elif "timeout" in error_str:
            print("\n⚠️  TIMEOUT ISSUE DETECTED")
            print("   Possible causes:")
            print("   - Slow network connection")
            print("   - API endpoint not reachable")
            print(f"   - Timeout too short ({openai_config.timeout}s)")
    
    # Alternative validation test
    print("\n5️⃣  Testing Alternative Validation (list models)\n")
    
    try:
        print("Attempting to list available models...")
        models = await client.models.list()
        model_ids = [m.id for m in models.data[:5]]  # First 5 models
        print(f"✅ Successfully listed models!")
        print(f"   Available models (first 5): {model_ids}")
        
        if openai_config.model in [m.id for m in models.data]:
            print(f"   ✅ Configured model '{openai_config.model}' is available")
        else:
            print(f"   ❌ Configured model '{openai_config.model}' NOT in available models")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
    
    # Test a simple completion
    print("\n6️⃣  Testing Simple Completion\n")
    
    try:
        print("Attempting a simple completion...")
        response = await client.chat.completions.create(
            model=openai_config.model,
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5,
            temperature=0
        )
        print(f"✅ Completion successful!")
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Completion failed: {e}")
    
    # Check environment variables
    print("\n7️⃣  Environment Check\n")
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
    print("Proxy settings:")
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}: {value[:50]}...")
        else:
            print(f"   {var}: Not set")
    
    # Close the client
    print("\n8️⃣  Cleanup\n")
    try:
        await client.close()
        print("✅ Client closed successfully")
    except Exception as e:
        print(f"⚠️  Error closing client: {e}")
    
    print("\n" + "=" * 80)
    print("VALIDATION DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_validation())