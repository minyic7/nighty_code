#!/usr/bin/env python3
"""
Simple Copilot Test Example
Tests the basic functionality of the Copilot module
"""

import asyncio
from src.copilot import InteractiveCopilot, CopilotWorkflow
from src.copilot.core.types import CopilotConfig
from src.llm import get_llm_manager, LLMProvider


async def test_interactive_copilot():
    """Test basic interactive copilot functionality"""
    print("\n=== Testing Interactive Copilot ===\n")
    
    # Create interactive copilot with default config
    copilot = InteractiveCopilot()
    
    # Initialize it
    await copilot.initialize()
    
    # Test sending a message through the workflow
    print("Testing basic query...")
    
    # Process through workflow
    if copilot.workflow:
        response = await copilot.workflow.process_message("What is Python?")
        if response:
            print(f"Copilot: {response.content[:300]}...")
    
    await copilot.shutdown()


async def test_copilot_workflow():
    """Test copilot workflow functionality"""
    print("\n=== Testing Copilot Workflow ===\n")
    
    # Get LLM manager and client
    llm_manager = await get_llm_manager()
    llm_client = llm_manager.get_client(LLMProvider.ANTHROPIC)
    
    # Create workflow with config
    config = CopilotConfig()
    workflow = CopilotWorkflow(
        config=config,
        llm_client=llm_client,
        mcp_manager=None  # Optional
    )
    
    # Initialize workflow
    await workflow.initialize()
    
    
    # Run workflow
    print("Running workflow...")
    response = await workflow.process_message("Explain the concept of recursion with a simple example")
    
    # Display results
    if response:
        print(f"\nAssistant Response:\n{response.content[:500]}...")


async def test_copilot_with_tools():
    """Test copilot with MCP tools"""
    print("\n=== Testing Copilot with Tools ===\n")
    
    # Create copilot with tools enabled
    config = CopilotConfig(
        enable_tools=True,
        enable_memory=True
    )
    copilot = InteractiveCopilot(config=config)
    
    # Initialize (this will set up MCP servers)
    await copilot.initialize()
    
    # Test a query that might use tools
    # Process through workflow
    if copilot.workflow:
        response = await copilot.workflow.process_message("List the Python files in the src directory")
        if response:
            print(f"Copilot: {response.content[:500]}...")
    
    await copilot.shutdown()


async def main():
    """Run all tests"""
    print("=" * 60)
    print("COPILOT MODULE TEST")
    print("=" * 60)
    
    try:
        # Run basic interactive test
        await test_interactive_copilot()
        
        # Run workflow test
        await test_copilot_workflow()
        
        # Run tools test
        await test_copilot_with_tools()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())