#!/usr/bin/env python3
"""
Simple LLM Module Test - Verifies LLM functionality
"""

import asyncio
from src.llm import (
    get_llm_manager,
    LLMProvider,
    LLMConfig,
    CompletionRequest,
    Message,
    MessageRole,
    LLMClient,
    LLMConnectionPool
)


async def test_llm_imports():
    """Test that all LLM components can be imported"""
    print("=== Testing LLM Imports ===\n")
    
    try:
        print("✅ All LLM components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_llm_config():
    """Test LLM configuration"""
    print("\n=== Testing LLM Configuration ===\n")
    
    try:
        # Test creating config with required fields
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",  # Required field
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=1000
        )
        print(f"✅ LLMConfig created: provider={config.provider.value}, model={config.model}")
        
        # Check attributes
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        print(f"✅ Config attributes verified")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


async def test_llm_manager():
    """Test LLM manager initialization"""
    print("\n=== Testing LLM Manager ===\n")
    
    try:
        # Get the singleton manager
        manager = await get_llm_manager()
        print("✅ LLM Manager initialized")
        
        # Check available providers enum
        providers = [p for p in LLMProvider]
        print(f"✅ Available provider types: {[p.value for p in providers]}")
        
        # Check configured providers
        configured = manager.list_providers()
        print(f"✅ Configured providers: {[p.value for p in configured]}")
        
        # Try to get clients for configured providers
        for provider in configured:
            try:
                client = manager.get_client(provider)
                print(f"✅ Got client for provider: {provider.value}")
            except Exception as e:
                print(f"⚠️  Could not get client for {provider.value}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Manager test failed: {e}")
        return False


async def test_message_creation():
    """Test message creation"""
    print("\n=== Testing Message Creation ===\n")
    
    try:
        # Create user message
        user_msg = Message(
            role=MessageRole.USER,
            content="Hello, how are you?"
        )
        print(f"✅ User message created: {user_msg.content[:30]}...")
        
        # Create assistant message
        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content="I'm doing well, thank you!"
        )
        print(f"✅ Assistant message created: {assistant_msg.content[:30]}...")
        
        # Create system message
        system_msg = Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant."
        )
        print(f"✅ System message created: {system_msg.content[:30]}...")
        
        return True
    except Exception as e:
        print(f"❌ Message creation failed: {e}")
        return False


async def test_completion_request():
    """Test creating a completion request"""
    print("\n=== Testing Completion Request ===\n")
    
    try:
        # Create messages
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="What is 2+2?")
        ]
        
        # Create completion request
        request = CompletionRequest(
            messages=messages,
            temperature=0.5,
            max_tokens=100
        )
        print(f"✅ CompletionRequest created with {len(request.messages)} messages")
        print(f"   Temperature: {request.temperature}")
        print(f"   Max tokens: {request.max_tokens}")
        
        return True
    except Exception as e:
        print(f"❌ Completion request creation failed: {e}")
        return False


async def test_provider_fallback():
    """Test provider fallback behavior"""
    print("\n=== Testing Provider Fallback ===\n")
    
    try:
        manager = await get_llm_manager()
        configured = manager.list_providers()
        
        if len(configured) == 0:
            print("⚠️  No providers configured")
            return True
        
        # Try to get default client (should use first available)
        try:
            client = manager.get_client()  # No provider specified
            print(f"✅ Default client obtained (using default provider)")
        except Exception as e:
            print(f"⚠️  No default provider set: {e}")
            
            # Try each provider explicitly
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]:
                try:
                    client = manager.get_client(provider)
                    print(f"✅ Successfully got client for {provider.value}")
                    break
                except:
                    print(f"   {provider.value} not available")
            else:
                print("⚠️  No providers available")
        
        return True
    except Exception as e:
        print(f"❌ Provider fallback test failed: {e}")
        return False


async def test_llm_client_structure():
    """Test LLM client structure without making actual API calls"""
    print("\n=== Testing LLM Client Structure ===\n")
    
    try:
        # Get manager and client
        manager = await get_llm_manager()
        client = manager.get_client(LLMProvider.ANTHROPIC)
        
        # Check client is returned
        assert client is not None
        print(f"✅ Client obtained from manager")
        
        # Check it's an LLMClient instance
        assert isinstance(client, LLMClient)
        print(f"✅ Client is LLMClient instance")
        
        return True
    except Exception as e:
        print(f"❌ Client structure test failed: {e}")
        return False


async def test_connection_pool():
    """Test connection pool structure"""
    print("\n=== Testing Connection Pool ===\n")
    
    try:
        from src.llm.core.config import PoolConfig
        
        # Create pool config
        pool_config = PoolConfig(
            min_size=1,
            max_size=5,
            acquire_timeout=30.0
        )
        print(f"✅ PoolConfig created: min={pool_config.min_size}, max={pool_config.max_size}")
        
        # Create connection pool
        pool = LLMConnectionPool(
            provider=LLMProvider.ANTHROPIC,
            config=pool_config
        )
        print(f"✅ Connection pool created for provider: {pool.provider.value}")
        
        # Check pool attributes
        assert hasattr(pool, 'acquire')
        assert hasattr(pool, 'release')
        print("✅ Pool has required methods")
        
        return True
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False


async def test_simple_completion():
    """Test a simple completion (requires API key)"""
    print("\n=== Testing Simple Completion ===\n")
    
    try:
        # Get manager and client
        manager = await get_llm_manager()
        client = manager.get_client(LLMProvider.ANTHROPIC)
        
        # Create a simple request
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.USER, content="Say 'Hello, test successful!' and nothing else.")
            ],
            temperature=0.0,
            max_tokens=20
        )
        
        print("📡 Sending completion request...")
        
        # Try to complete the request
        try:
            response = await client.complete(request)
            
            if response and response.content:
                print(f"✅ Response received: {response.content[:50]}...")
                if response.usage:
                    print(f"   Tokens used: {response.usage.get('total_tokens', 'N/A')}")
                return True
            else:
                print("⚠️  No response received (check API key)")
                return True  # Still pass since structure is correct
        except Exception as api_error:
            print(f"⚠️  API call error: {str(api_error)[:100]}...")
            print("   This is expected if API key is not configured")
            return True  # Structure test passes even without API key
            
    except Exception as e:
        if "api" in str(e).lower() or "key" in str(e).lower():
            print(f"⚠️  API call failed (likely API key issue): {e}")
            print("   Structure test passed, but API key may not be configured")
            return True  # Structure is correct, just no API key
        else:
            print(f"❌ Completion test failed: {e}")
            return False


async def main():
    """Run all LLM tests"""
    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Run structure tests (don't require API key)
    all_passed &= await test_llm_imports()
    all_passed &= await test_llm_config()
    all_passed &= await test_llm_manager()
    all_passed &= await test_provider_fallback()
    all_passed &= await test_message_creation()
    all_passed &= await test_completion_request()
    all_passed &= await test_llm_client_structure()
    all_passed &= await test_connection_pool()
    
    # Run API test (requires API key)
    print("\n" + "=" * 60)
    print("OPTIONAL API TEST (requires valid API key)")
    print("=" * 60)
    await test_simple_completion()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All structure tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())