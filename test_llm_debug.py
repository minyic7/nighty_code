#!/usr/bin/env python3
"""
Comprehensive LLM Debug Test
Shows all configuration, pool, client, and manager details
"""

import asyncio
import json
from pprint import pprint
from src.llm import get_llm_manager, LLMProvider, Message, MessageRole


async def debug_test():
    """Debug test with full configuration inspection"""
    
    print("=" * 80)
    print("LLM COMPREHENSIVE DEBUG TEST")
    print("=" * 80 + "\n")
    
    # Get the LLM manager
    print("📦 Initializing LLM Manager...")
    manager = await get_llm_manager()
    print("✅ Manager initialized\n")
    
    # ========== SECTION 1: Configuration Manager ==========
    print("=" * 80)
    print("1️⃣  CONFIGURATION MANAGER DETAILS")
    print("=" * 80)
    
    if hasattr(manager, '_config_manager'):
        config_mgr = manager._config_manager
        
        # Global config
        print("\n📋 Global Configuration:")
        if hasattr(config_mgr, 'global_config'):
            gc = config_mgr.global_config
            print(f"  • Default Provider: {gc.default_provider.value if gc.default_provider else 'None'}")
            print(f"  • Enable Logging: {gc.enable_logging}")
            print(f"  • Log Level: {gc.log_level}")
            print(f"  • Metrics Enabled: {gc.metrics_enabled}")
            
            # Pool config
            print(f"\n  📊 Pool Configuration:")
            pc = gc.pool_config
            print(f"    - Min Size: {pc.min_size}")
            print(f"    - Max Size: {pc.max_size}")
            print(f"    - Acquire Timeout: {pc.acquire_timeout}s")
            print(f"    - Idle Timeout: {pc.idle_timeout}s")
            print(f"    - Max Lifetime: {pc.max_lifetime}s")
            print(f"    - Retry on Error: {pc.retry_on_error}")
            print(f"    - Health Check Interval: {pc.health_check_interval}s")
            print(f"    - Enable Metrics: {pc.enable_metrics}")
        
        # Provider configs
        print("\n📋 Provider Configurations:")
        if hasattr(config_mgr, 'provider_configs'):
            for i, config in enumerate(config_mgr.provider_configs):
                print(f"\n  Config #{i+1} ({config.provider.value}):")
                print(f"    • Provider: {config.provider.value}")
                print(f"    • Model: {config.model}")
                print(f"    • API Key: {'*' * 20}...{config.api_key[-10:] if config.api_key else 'None'}")
                print(f"    • Base URL: {config.base_url or 'Default'}")
                print(f"    • Temperature: {config.temperature}")
                print(f"    • Max Tokens: {config.max_tokens}")
                print(f"    • Timeout: {config.timeout}s")
                print(f"    • Max Retries: {config.max_retries}")
                
                if config.rate_limit_config:
                    rl = config.rate_limit_config
                    print(f"    • Rate Limits:")
                    print(f"      - Requests/min: {rl.requests_per_minute}")
                    print(f"      - Input tokens/min: {rl.input_tokens_per_minute}")
                    print(f"      - Output tokens/min: {rl.output_tokens_per_minute}")
    
    # ========== SECTION 2: Manager State ==========
    print("\n" + "=" * 80)
    print("2️⃣  MANAGER STATE")
    print("=" * 80)
    
    # Providers
    configured_providers = manager.list_providers()
    print(f"\n🔧 Configured Providers: {[p.value for p in configured_providers]}")
    
    # Clients dictionary
    if hasattr(manager, '_clients'):
        print(f"\n📦 Clients Dictionary (_clients):")
        for provider, client in manager._clients.items():
            print(f"  • {provider.value}: {type(client).__name__} (id: {id(client)})")
    
    # Pools dictionary
    if hasattr(manager, '_pools'):
        print(f"\n🏊 Pools Dictionary (_pools):")
        for provider, pool in manager._pools.items():
            print(f"  • {provider.value}: {type(pool).__name__} (id: {id(pool)})")
            
            # Pool details
            if hasattr(pool, '_configs'):
                print(f"    - Configurations in pool: {len(pool._configs)}")
                for j, cfg in enumerate(pool._configs):
                    print(f"      Config {j+1}: {cfg.model} (provider: {cfg.provider.value})")
            
            if hasattr(pool, '_clients'):
                print(f"    - Client instances in pool: {len(pool._clients)}")
            
            if hasattr(pool, '_semaphore'):
                print(f"    - Semaphore limit: {pool._semaphore._value}/{pool._semaphore._initial_value}")
    
    # ========== SECTION 3: Client Details ==========
    print("\n" + "=" * 80)
    print("3️⃣  CLIENT DETAILS")
    print("=" * 80)
    
    for provider in configured_providers:
        print(f"\n🤖 {provider.value.upper()} Client:")
        try:
            client = manager.get_client(provider)
            print(f"  • Type: {type(client).__name__}")
            print(f"  • ID: {id(client)}")
            
            if hasattr(client, 'pool'):
                pool = client.pool
                print(f"  • Pool ID: {id(pool)}")
                print(f"  • Pool Type: {type(pool).__name__}")
                
                if hasattr(pool, '_configs'):
                    print(f"  • Configs in pool: {len(pool._configs)}")
                
                if hasattr(pool, 'provider'):
                    print(f"  • Pool Provider: {pool.provider.value}")
            
            if hasattr(client, 'instructor_client'):
                print(f"  • Has Instructor: {client.instructor_client is not None}")
            
            # Check if same instance on second call
            client2 = manager.get_client(provider)
            print(f"  • Singleton check: {'✅ Same instance' if client is client2 else '❌ Different instance'}")
            
        except Exception as e:
            print(f"  ❌ Error getting client: {e}")
    
    # ========== SECTION 4: Test Query ==========
    print("\n" + "=" * 80)
    print("4️⃣  TEST QUERY")
    print("=" * 80)
    
    # Get default client
    print("\n🎯 Using default provider for test query...")
    client = manager.get_client()  # Uses default
    
    # Simple question
    question = "What is 2 + 2?"
    print(f"\n❓ Question: {question}")
    
    try:
        import time
        start_time = time.time()
        
        # Make the request
        messages = [Message(role=MessageRole.USER, content=question)]
        response = await client.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=20
        )
        
        elapsed = time.time() - start_time
        
        print(f"💬 Answer: {response.content}")
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"🤖 Model used: {response.model}")
        print(f"🏢 Provider: {response.provider.value}")
        
        if response.usage:
            print(f"📊 Token usage: {response.usage}")
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== SECTION 5: Summary ==========
    print("\n" + "=" * 80)
    print("5️⃣  SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Active Providers: {len(configured_providers)}")
    print(f"✅ Total Clients: {len(manager._clients) if hasattr(manager, '_clients') else 0}")
    print(f"✅ Total Pools: {len(manager._pools) if hasattr(manager, '_pools') else 0}")
    
    if hasattr(manager, '_config_manager') and hasattr(manager._config_manager, 'provider_configs'):
        total_configs = len(manager._config_manager.provider_configs)
        print(f"✅ Total Configurations: {total_configs}")
        
        # Count by provider
        by_provider = {}
        for cfg in manager._config_manager.provider_configs:
            provider = cfg.provider.value
            by_provider[provider] = by_provider.get(provider, 0) + 1
        
        for provider, count in by_provider.items():
            print(f"  • {provider}: {count} config(s)")
    
    print("\n" + "=" * 80)
    print("🎉 Debug test completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(debug_test())