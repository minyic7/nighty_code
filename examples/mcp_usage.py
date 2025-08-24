#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Module Usage Examples

This script demonstrates the MCP module capabilities:
- Filesystem operations (read, write, search)
- Tool management and execution
- Server/client communication
- Resource discovery
- Prompt templates
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any

from src.mcp import (
    FilesystemServer,
    MCPManager,
    ToolCall,
    ToolResult,
    Resource,
    Prompt,
    MCPException
)


# Example 1: Basic filesystem operations
async def filesystem_operations_example():
    """Demonstrate basic filesystem operations through MCP"""
    print("\n=== Filesystem Operations Example ===")
    
    # Initialize filesystem server
    server = FilesystemServer(
        allowed_directories=[Path.cwd()],  # Only allow current directory
        read_only=False
    )
    
    await server.initialize()
    
    # List available tools
    tools = await server.list_tools()
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Read a file
    try:
        read_call = ToolCall(
            name="read_file",
            arguments={
                "path": "README.md"
            }
        )
        
        result = await server.call_tool(read_call)
        if result.is_success:
            content = result.content[0].text if result.content else ""
            print(f"\nFile content preview: {content[:200]}...")
        else:
            print(f"Read failed: {result.error}")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    # Search for files
    try:
        search_call = ToolCall(
            name="search_files",
            arguments={
                "pattern": "*.py",
                "directory": "src"
            }
        )
        
        result = await server.call_tool(search_call)
        if result.is_success:
            files = result.content[0].text if result.content else ""
            print(f"\nPython files found:\n{files[:500]}...")
    except Exception as e:
        print(f"Error searching files: {e}")
    
    await server.cleanup()


# Example 2: Writing and modifying files
async def file_modification_example():
    """Demonstrate file creation and modification"""
    print("\n=== File Modification Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd() / "examples"],
        read_only=False
    )
    
    await server.initialize()
    
    # Create a new file
    test_file = "examples/test_mcp_file.txt"
    
    try:
        # Write initial content
        write_call = ToolCall(
            name="write_file",
            arguments={
                "path": test_file,
                "content": "Hello from MCP!\nThis is a test file.\n"
            }
        )
        
        result = await server.call_tool(write_call)
        if result.is_success:
            print(f"File created: {test_file}")
        
        # Append to file
        append_call = ToolCall(
            name="append_to_file",
            arguments={
                "path": test_file,
                "content": "Additional line added via MCP.\n"
            }
        )
        
        result = await server.call_tool(append_call)
        if result.is_success:
            print(f"Content appended to {test_file}")
        
        # Read back the file
        read_call = ToolCall(
            name="read_file",
            arguments={"path": test_file}
        )
        
        result = await server.call_tool(read_call)
        if result.is_success:
            content = result.content[0].text if result.content else ""
            print(f"\nFinal file content:\n{content}")
        
        # Clean up - delete the test file
        delete_call = ToolCall(
            name="delete_file",
            arguments={"path": test_file}
        )
        await server.call_tool(delete_call)
        print(f"\nTest file cleaned up")
        
    except Exception as e:
        print(f"Error in file operations: {e}")
    
    await server.cleanup()


# Example 3: Using grep for code search
async def code_search_example():
    """Search for patterns in code using grep"""
    print("\n=== Code Search Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd()],
        enable_grep=True
    )
    
    await server.initialize()
    
    # Search for class definitions
    try:
        grep_call = ToolCall(
            name="grep",
            arguments={
                "pattern": "^class \\w+",
                "directory": "src/mcp",
                "file_pattern": "*.py"
            }
        )
        
        result = await server.call_tool(grep_call)
        if result.is_success:
            matches = result.content[0].text if result.content else ""
            print(f"Class definitions found:\n{matches[:1000]}...")
    except Exception as e:
        print(f"Error in grep search: {e}")
    
    # Search for async functions
    try:
        grep_call = ToolCall(
            name="grep",
            arguments={
                "pattern": "async def",
                "directory": "src/mcp",
                "file_pattern": "*.py",
                "max_results": 10
            }
        )
        
        result = await server.call_tool(grep_call)
        if result.is_success:
            matches = result.content[0].text if result.content else ""
            print(f"\nAsync functions found:\n{matches}")
    except Exception as e:
        print(f"Error in grep search: {e}")
    
    await server.cleanup()


# Example 4: Resource management
async def resource_management_example():
    """Demonstrate resource discovery and management"""
    print("\n=== Resource Management Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd()]
    )
    
    await server.initialize()
    
    # List available resources
    resources = await server.list_resources()
    print(f"Available resources: {len(resources)}")
    
    for resource in resources[:5]:  # Show first 5
        print(f"  - {resource.name}: {resource.uri}")
        if resource.description:
            print(f"    Description: {resource.description}")
    
    # Read a specific resource
    if resources:
        try:
            resource_content = await server.read_resource(resources[0].uri)
            if resource_content:
                print(f"\nResource content preview: {resource_content[:200]}...")
        except Exception as e:
            print(f"Error reading resource: {e}")
    
    await server.cleanup()


# Example 5: Prompt templates
async def prompt_templates_example():
    """Demonstrate using prompt templates"""
    print("\n=== Prompt Templates Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd()]
    )
    
    await server.initialize()
    
    # List available prompts
    prompts = await server.list_prompts()
    print(f"Available prompts: {len(prompts)}")
    
    for prompt in prompts:
        print(f"  - {prompt.name}")
        if prompt.description:
            print(f"    Description: {prompt.description}")
        if prompt.arguments:
            print(f"    Arguments: {[arg.name for arg in prompt.arguments]}")
    
    # Get a specific prompt
    if prompts:
        try:
            prompt_result = await server.get_prompt(
                prompts[0].name,
                arguments={}  # Add any required arguments
            )
            
            if prompt_result:
                print(f"\nPrompt content:\n{prompt_result[:500]}...")
        except Exception as e:
            print(f"Error getting prompt: {e}")
    
    await server.cleanup()


# Example 6: MCPManager for multiple servers
async def manager_example():
    """Demonstrate managing multiple MCP servers"""
    print("\n=== MCP Manager Example ===")
    
    manager = MCPManager()
    
    # Register filesystem server
    fs_server = FilesystemServer(
        allowed_directories=[Path.cwd()],
        server_name="main_fs"
    )
    
    await manager.register_server("filesystem", fs_server)
    
    # You could register additional servers here
    # await manager.register_server("database", DatabaseServer())
    # await manager.register_server("api", APIServer())
    
    # List all registered servers
    servers = manager.list_servers()
    print(f"Registered servers: {servers}")
    
    # Execute tool on specific server
    try:
        result = await manager.execute_tool(
            server_name="filesystem",
            tool_call=ToolCall(
                name="list_directory",
                arguments={"path": "."}
            )
        )
        
        if result.is_success:
            print(f"\nDirectory listing successful")
    except Exception as e:
        print(f"Error executing tool: {e}")
    
    # Discover all tools across servers
    all_tools = await manager.discover_tools()
    print(f"\nTotal tools available: {len(all_tools)}")
    for server_name, tools in all_tools.items():
        print(f"  {server_name}: {len(tools)} tools")
    
    await manager.cleanup()


# Example 7: Advanced file operations
async def advanced_file_operations():
    """Demonstrate advanced file operations"""
    print("\n=== Advanced File Operations Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd() / "examples"],
        max_file_size=10 * 1024 * 1024,  # 10MB limit
        enable_hidden_files=False
    )
    
    await server.initialize()
    
    # Get file metadata
    try:
        metadata_call = ToolCall(
            name="get_file_info",
            arguments={"path": "examples/llm_usage.py"}
        )
        
        result = await server.call_tool(metadata_call)
        if result.is_success:
            info = result.content[0].text if result.content else ""
            print(f"File metadata:\n{info}")
    except Exception as e:
        print(f"Error getting metadata: {e}")
    
    # Compare files
    try:
        diff_call = ToolCall(
            name="diff_files",
            arguments={
                "file1": "examples/llm_usage.py",
                "file2": "examples/mcp_usage.py",
                "context_lines": 3
            }
        )
        
        result = await server.call_tool(diff_call)
        if result.is_success:
            diff = result.content[0].text if result.content else ""
            print(f"\nFile differences (first 500 chars):\n{diff[:500]}...")
    except Exception as e:
        print(f"Error comparing files: {e}")
    
    # Move/rename file (demonstration only)
    test_file = "examples/test_rename.txt"
    new_name = "examples/test_renamed.txt"
    
    try:
        # Create test file
        await server.call_tool(ToolCall(
            name="write_file",
            arguments={
                "path": test_file,
                "content": "Test file for renaming"
            }
        ))
        
        # Rename file
        move_call = ToolCall(
            name="move_file",
            arguments={
                "source": test_file,
                "destination": new_name
            }
        )
        
        result = await server.call_tool(move_call)
        if result.is_success:
            print(f"\nFile renamed from {test_file} to {new_name}")
        
        # Clean up
        await server.call_tool(ToolCall(
            name="delete_file",
            arguments={"path": new_name}
        ))
        
    except Exception as e:
        print(f"Error in file operations: {e}")
    
    await server.cleanup()


# Example 8: Batch operations
async def batch_operations_example():
    """Demonstrate batch file operations"""
    print("\n=== Batch Operations Example ===")
    
    server = FilesystemServer(
        allowed_directories=[Path.cwd() / "examples"]
    )
    
    await server.initialize()
    
    # Create multiple test files
    test_files = [
        ("examples/batch_test_1.txt", "Content 1"),
        ("examples/batch_test_2.txt", "Content 2"),
        ("examples/batch_test_3.txt", "Content 3"),
    ]
    
    try:
        # Batch create
        for filepath, content in test_files:
            await server.call_tool(ToolCall(
                name="write_file",
                arguments={"path": filepath, "content": content}
            ))
        print(f"Created {len(test_files)} test files")
        
        # Batch read
        contents = []
        for filepath, _ in test_files:
            result = await server.call_tool(ToolCall(
                name="read_file",
                arguments={"path": filepath}
            ))
            if result.is_success and result.content:
                contents.append(result.content[0].text)
        
        print(f"Read {len(contents)} files successfully")
        
        # Batch delete
        for filepath, _ in test_files:
            await server.call_tool(ToolCall(
                name="delete_file",
                arguments={"path": filepath}
            ))
        print(f"Cleaned up {len(test_files)} test files")
        
    except Exception as e:
        print(f"Error in batch operations: {e}")
    
    await server.cleanup()


async def main():
    """Run all MCP examples"""
    print("=" * 60)
    print("MCP MODULE USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    await filesystem_operations_example()
    await file_modification_example()
    await code_search_example()
    await resource_management_example()
    await prompt_templates_example()
    await manager_example()
    await advanced_file_operations()
    await batch_operations_example()
    
    print("\n" + "=" * 60)
    print("All MCP examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())