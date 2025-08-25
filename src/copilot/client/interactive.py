# src/copilot/client/interactive.py
"""
Interactive REPL client for the copilot
"""

import asyncio
import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

from ..core.workflow import CopilotWorkflow
from ..core.types import CopilotConfig, AssistantResponse
from src.llm import get_llm_manager, LLMProvider
from src.mcp import MCPManager, BaseMCPServer
from src.mcp.servers.filesystem import FilesystemServer

logger = logging.getLogger(__name__)


class InteractiveCopilot:
    """Interactive REPL interface for the copilot"""
    
    def __init__(self, config: Optional[CopilotConfig] = None):
        self.config = config or CopilotConfig()
        self.workflow: Optional[CopilotWorkflow] = None
        self.mcp_manager: Optional[MCPManager] = None
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.running = False
    
    async def initialize(self):
        """Initialize the copilot"""
        print("🚀 Initializing Copilot...")
        
        try:
            # Initialize LLM
            llm_manager = await get_llm_manager()
            
            # Use provider from config or default
            provider = LLMProvider(self.config.llm_provider)
            llm_client = llm_manager.get_client(provider)
            
            # Initialize MCP Manager
            self.mcp_manager = MCPManager()
            
            # Add filesystem server by default
            fs_server = FilesystemServer(
                root_path=Path.cwd()  # Use current directory as root
            )
            await fs_server.initialize()
            self.mcp_manager.register_server("filesystem", fs_server)
            
            # Create workflow
            self.workflow = CopilotWorkflow(
                config=self.config,
                llm_client=llm_client,
                mcp_manager=self.mcp_manager
            )
            await self.workflow.initialize()
            
            print("✅ Copilot initialized successfully!")
            print(f"📁 Working directory: {Path.cwd()}")
            print(f"🔧 Available MCP servers: {list(self.mcp_manager.get_servers().keys())}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
    
    async def run(self):
        """Run the interactive REPL"""
        await self.initialize()
        
        self.running = True
        print("💬 Interactive Copilot Session")
        print("=" * 50)
        print("Type 'help' for commands, 'exit' to quit")
        print("=" * 50)
        print()
        
        while self.running:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    await self.shutdown()
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                elif user_input.lower() == 'tools':
                    await self._show_tools()
                    continue
                elif user_input.lower() == 'memory':
                    await self._show_memory()
                    continue
                
                # Process message through workflow
                print("\n🤔 Thinking...")
                response = await self.workflow.process_message(user_input, self.session_id)
                
                # Display response
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'exit' to quit properly.")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Error in interactive loop: {e}", exc_info=True)
        
        print("\n👋 Goodbye!")
    
    async def _get_user_input(self) -> str:
        """Get input from user"""
        # Use asyncio-compatible input
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input("\n🧑 You: "))
    
    def _display_response(self, response: AssistantResponse):
        """Display the assistant's response"""
        print(f"\n🤖 Assistant: {response.content}")
        
        if self.config.verbose:
            # Show thoughts if verbose
            if response.thoughts:
                print("\n💭 Thoughts:")
                for thought in response.thoughts:
                    print(f"  • {thought.type.value}: {thought.content[:100]}...")
            
            # Show actions if any
            if response.actions_taken:
                print("\n🔧 Actions:")
                for action in response.actions_taken:
                    status_emoji = "✅" if action.status == "success" else "❌"
                    print(f"  {status_emoji} {action.tool_name}: {action.status}")
            
            # Show confidence
            print(f"\n📊 Confidence: {response.confidence:.1%}")
        
        if response.needs_clarification:
            print("\n❓ I need more information to help you better.")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  help    - Show this help message")
        print("  status  - Show copilot status")
        print("  tools   - List available tools")
        print("  memory  - Show memory statistics")
        print("  clear   - Clear the screen")
        print("  exit    - Exit the copilot")
        print("\n💡 Tips:")
        print("  • Ask questions about your codebase")
        print("  • Request file operations (read, search, list)")
        print("  • Get help with coding tasks")
        print("  • Ask for explanations and analysis")
    
    async def _show_status(self):
        """Show current status"""
        if not self.workflow:
            print("❌ Workflow not initialized")
            return
        
        print("\n📊 Copilot Status:")
        print(f"  Session ID: {self.session_id}")
        print(f"  Config:")
        print(f"    • Provider: {self.config.llm_provider}")
        print(f"    • Memory: {'Enabled' if self.config.enable_memory else 'Disabled'}")
        print(f"    • Guardrails: {'Enabled' if self.config.enable_guardrails else 'Disabled'}")
        print(f"    • Auto-execute: {'Yes' if self.config.auto_execute_tools else 'No'}")
    
    async def _show_tools(self):
        """Show available tools"""
        if not self.workflow:
            print("❌ Workflow not initialized")
            return
        
        tools = self.workflow.tool_router.get_available_tools()
        
        print("\n🔧 Available Tools:")
        for server, tool_list in tools.items():
            print(f"\n  {server} server:")
            for tool in tool_list:
                print(f"    • {tool}")
    
    async def _show_memory(self):
        """Show memory statistics"""
        if not self.workflow:
            print("❌ Workflow not initialized")
            return
        
        # This would need access to the current state
        # For now, show config
        print("\n🧠 Memory Configuration:")
        print(f"  Enabled: {self.config.enable_memory}")
        print(f"  Persistence: {self.config.memory_persistence}")
        if self.config.memory_file:
            print(f"  File: {self.config.memory_file}")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        print("💬 Interactive Copilot Session")
        print("=" * 50)
    
    async def shutdown(self):
        """Shutdown the copilot"""
        print("\n🔄 Shutting down...")
        self.running = False
        
        # Save memories if enabled
        if self.workflow and self.config.memory_persistence:
            # Would need to implement memory saving
            pass
        
        # Close MCP servers
        if self.mcp_manager:
            servers = self.mcp_manager.get_servers()
            for server in servers.values():
                if hasattr(server, 'close'):
                    await server.close()


async def main():
    """Main entry point for interactive copilot"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = CopilotConfig(
        verbose=False,  # Set to True for detailed output
        enable_memory=True,
        enable_guardrails=True,
        auto_execute_tools=True
    )
    
    # Create and run copilot
    copilot = InteractiveCopilot(config)
    
    try:
        await copilot.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())