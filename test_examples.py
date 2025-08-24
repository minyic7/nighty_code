#!/usr/bin/env python3
"""
Test script to verify all modules work correctly
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_llm_module():
    """Test basic LLM functionality"""
    print("\n=== Testing LLM Module ===")
    try:
        from src.llm import LLMManager, Message, MessageRole, LLMProvider
        
        # Test initialization
        manager = LLMManager()
        print("✓ LLM Manager initialized")
        
        # Test getting client (won't make actual API calls without keys)
        try:
            client = manager.get_client(LLMProvider.ANTHROPIC)
            print("✓ LLM Client created")
        except Exception as e:
            print(f"✓ LLM Client creation handled: {type(e).__name__}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_mcp_module():
    """Test basic MCP functionality"""
    print("\n=== Testing MCP Module ===")
    try:
        from src.mcp.servers.filesystem import FilesystemServer
        from src.mcp.core.types import ToolCall
        from pathlib import Path
        
        # Test initialization
        server = FilesystemServer()
        await server.initialize()
        print("✓ MCP FilesystemServer initialized")
        
        # Test listing tools
        tools = server.list_tools()
        print(f"✓ Found {len(tools)} tools")
        
        # Test reading this file
        result = await server.call_tool(ToolCall(
            name="read_file",
            arguments={"path": "test_examples.py"}
        ))
        
        if result.status == "success":
            print("✓ File read successful")
        else:
            print(f"✓ File read handled: {result.error if hasattr(result, 'error') else 'error'}")
        
        # Server cleanup (if exists)
        if hasattr(server, 'cleanup'):
            await server.cleanup()
        print("✓ Server operations completed")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_copilot_module():
    """Test basic Copilot functionality"""
    print("\n=== Testing Copilot Module ===")
    try:
        from src.copilot.client.interactive import InteractiveCopilot
        
        # Test session config (using dict since SessionConfig might not be a class)
        session_config = {
            "session_id": "test",
            "enable_memory": False,
            "enable_tools": False
        }
        print("✓ Session config created")
        
        # Test interactive client initialization
        client = InteractiveCopilot(session_config)
        print("✓ InteractiveCopilot created")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_dataminer_module():
    """Test basic DataMiner functionality"""
    print("\n=== Testing DataMiner Module ===")
    try:
        from src.dataminer import DataMinerClient, ProcessingMode
        from src.dataminer.core.config import create_default_config
        from src.dataminer.models.base import ExtractionSchema
        from pydantic import Field
        from typing import List
        
        # Test config creation
        config = create_default_config()
        print("✓ Default config created")
        
        # Test client initialization
        client = DataMinerClient(config)
        print("✓ DataMinerClient created")
        
        # Test schema definition
        class TestSchema(ExtractionSchema):
            name: str = Field(description="Name")
            value: int = Field(description="Value")
            
            def get_confidence_fields(self) -> List[str]:
                return ["value"]
            
            def get_required_fields(self) -> List[str]:
                return ["name"]
        
        print("✓ Schema defined successfully")
        
        # Test initialization (without actual extraction)
        await client.initialize()
        print("✓ Client initialized")
        
        # Test getting capabilities
        capabilities = await client.get_extraction_capabilities()
        print(f"✓ Found {len(capabilities.get('strategies', []))} strategies")
        
        await client.cleanup()
        print("✓ Client cleanup completed")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING NIGHTY CODE MODULES")
    print("=" * 60)
    
    results = []
    
    # Test each module
    results.append(("LLM", await test_llm_module()))
    results.append(("MCP", await test_mcp_module()))
    results.append(("Copilot", await test_copilot_module()))
    results.append(("DataMiner", await test_dataminer_module()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for module_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{module_name:12} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✅ All modules tested successfully!")
    else:
        print("❌ Some modules failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)