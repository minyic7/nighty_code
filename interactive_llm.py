#!/usr/bin/env python3
"""
Interactive LLM session with pool status monitoring.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import (
    LLMManager,
    get_llm_manager,
    LLMProvider,
    Message,
    MessageRole,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveLLMSession:
    """Interactive session for playing with LLM client."""
    
    def __init__(self):
        self.manager: Optional[LLMManager] = None
        self.client = None
        self.history = []
        self.system_message = "You are a helpful assistant."
        
    async def initialize(self):
        """Initialize the LLM manager and client."""
        logger.info("Initializing LLM Manager...")
        self.manager = await get_llm_manager()
        
        # Pre-initialize all clients in the pool
        logger.info("Pre-initializing all clients in the pool...")
        pool = self.manager._pools[LLMProvider.ANTHROPIC]
        
        # Get the number of configured API keys
        num_configs = len(pool.configs)
        logger.info(f"Found {num_configs} API key configurations")
        
        # Expand pool to include all configured clients
        while len(pool._clients) < num_configs:
            logger.info(f"Creating client {len(pool._clients) + 1}/{num_configs}...")
            await pool._create_client()
        
        self.client = self.manager.get_client(LLMProvider.ANTHROPIC)
        logger.info(f"✓ LLM Manager initialized with {len(pool._clients)} clients ready")
        self.show_pool_status()
        
    def show_pool_status(self):
        """Display current pool status."""
        if not self.manager:
            return
            
        print("\n" + "="*60)
        print("POOL STATUS")
        print("="*60)
        
        status = self.manager.get_status()
        for provider_name, pool_status in status.items():
            print(f"\nProvider: {provider_name}")
            print(f"  Total API keys configured: {pool_status['total_configs']}")
            print(f"  Active clients in pool: {pool_status['total_clients']}")
            print(f"  Available for use: {pool_status['available_clients']}")
            print(f"  Currently in use: {pool_status['in_use_clients']}")
            
            metrics = pool_status['metrics']
            print(f"\nMetrics:")
            print(f"  Total requests: {metrics['total_requests']}")
            print(f"  Failed requests: {metrics['failed_requests']}")
            print(f"  Pool exhausted count: {metrics['pool_exhausted_count']}")
            print(f"  Average wait time: {metrics['average_wait_time']:.4f}s")
            
            # Show individual client status with API key index
            print(f"\nClient Details:")
            pool = self.manager._pools.get(LLMProvider(provider_name))
            for i, client in enumerate(pool_status['clients'], 1):
                # Find which API key index this client is using
                api_key_index = "?"
                if pool:
                    for idx, c in enumerate(pool._clients):
                        if c._client_id == client['client_id']:
                            # Find which config this client uses
                            for cfg_idx, cfg in enumerate(pool.configs):
                                if cfg.api_key == c.config.api_key:
                                    api_key_index = str(cfg_idx + 1)
                                    break
                            break
                
                status_icon = "✓" if client['is_available'] else "✗"
                in_use_icon = "🔒" if client['in_use'] else "🔓"
                print(f"  {status_icon} Client {i} (API Key #{api_key_index}): {client['client_id']} {in_use_icon}")
                print(f"      Requests: {client['total_requests']}, Errors: {client['error_count']}")
        
        print("="*60 + "\n")
    
    def show_help(self):
        """Display help information."""
        print("\n" + "="*60)
        print("INTERACTIVE LLM SESSION - COMMANDS")
        print("="*60)
        print()
        print("Commands:")
        print("  /help           - Show this help message")
        print("  /status         - Show pool status")
        print("  /clear          - Clear conversation history")
        print("  /history        - Show conversation history")
        print("  /system <msg>   - Set system message")
        print("  /stream <msg>   - Stream a response")
        print("  /tokens <text>  - Count tokens in text")
        print("  /model          - Show current model info")
        print("  /metrics        - Show detailed metrics")
        print("  /clients        - Show all client details")
        print("  /parallel <n>   - Send n parallel requests to test load balancing")
        print("  /exit or /quit  - Exit the session")
        print()
        print("Or just type your message to chat with Claude!")
        print("="*60 + "\n")
    
    async def count_tokens(self, text: str):
        """Count tokens in the given text."""
        from llm.middleware.token_calculator import TokenCalculatorFactory
        
        calculator = TokenCalculatorFactory.create_calculator(
            "anthropic",
            "claude-3-5-haiku-20241022"
        )
        
        tokens = calculator.count_tokens(text)
        print(f"\nToken count: {tokens}")
        print(f"Estimated characters per token: {len(text) / max(1, tokens):.2f}")
        
    async def chat(self, user_input: str, stream: bool = False):
        """Send a message to the LLM."""
        try:
            # Add to history
            self.history.append(Message(MessageRole.USER, user_input))
            
            # Prepare messages
            messages = []
            if self.system_message:
                messages.append(Message(MessageRole.SYSTEM, self.system_message))
            messages.extend(self.history)
            
            # Show pool status before request
            print("\n--- Pre-request Pool Status ---")
            status = self.manager.get_status()
            for provider, pool in status.items():
                print(f"{provider}: {pool['available_clients']} available, {pool['in_use_clients']} in use")
            
            if stream:
                print("\n🤖 Claude (streaming): ", end="", flush=True)
                full_response = ""
                async for chunk in self.client.stream_complete(messages, max_tokens=1000):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print("\n")
                self.history.append(Message(MessageRole.ASSISTANT, full_response))
            else:
                print("\n🤖 Claude is thinking...")
                response = await self.client.complete(messages, max_tokens=1000)
                print(f"\n🤖 Claude: {response.content}")
                
                # Show token usage
                if response.usage:
                    print(f"\n📊 Usage: {response.usage}")
                
                self.history.append(Message(MessageRole.ASSISTANT, response.content))
            
            # Show pool status after request
            print("\n--- Post-request Pool Status ---")
            status = self.manager.get_status()
            for provider, pool in status.items():
                print(f"{provider}: {pool['available_clients']} available, {pool['in_use_clients']} in use")
                print(f"Total requests so far: {pool['metrics']['total_requests']}")
            
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"\n❌ Error: {e}")
    
    async def show_metrics(self):
        """Show detailed metrics including rate limiter status."""
        print("\n" + "="*60)
        print("DETAILED METRICS")
        print("="*60)
        
        # Get pool status
        status = self.manager.get_status()
        
        for provider_name, pool_status in status.items():
            print(f"\n{provider_name.upper()} Provider:")
            
            # Show rate limiter status for each client
            for client_info in pool_status['clients']:
                print(f"\n  Client: {client_info['client_id']}")
                
                # Try to get rate limit status from the actual client
                pool = self.manager._pools.get(LLMProvider(provider_name))
                if pool:
                    for client in pool._clients:
                        if client._client_id == client_info['client_id']:
                            rl_status = client.get_rate_limit_status()
                            if rl_status:
                                print(f"    Rate Limit Status:")
                                print(f"      - Available requests: {rl_status['available']['requests']:.1f}")
                                print(f"      - Available input tokens: {rl_status['available']['input_tokens']:.0f}")
                                print(f"      - Available output tokens: {rl_status['available']['output_tokens']:.0f}")
                                print(f"      - Total requests handled: {rl_status['metrics']['total_requests']}")
                                print(f"      - Accepted: {rl_status['metrics']['accepted_requests']}")
                                print(f"      - Rejected: {rl_status['metrics']['rejected_requests']}")
                            break
        
        print("="*60 + "\n")
    
    async def test_parallel_requests(self, num_requests: int):
        """Test parallel requests to see load balancing in action."""
        import asyncio
        
        print(f"\n🚀 Sending {num_requests} parallel requests...")
        print("Watch how the pool distributes load across clients!\n")
        
        # Show initial status
        print("Initial pool state:")
        status = self.manager.get_status()
        for provider, pool in status.items():
            print(f"  {provider}: {pool['available_clients']} available, {pool['in_use_clients']} in use")
        
        async def make_request(i: int):
            """Make a single request."""
            try:
                messages = [
                    Message(MessageRole.USER, f"Say 'Response {i}' and nothing else")
                ]
                response = await self.client.complete(messages, max_tokens=20)
                return f"Request {i}: {response.content[:30]}"
            except Exception as e:
                return f"Request {i}: Error - {str(e)[:50]}"
        
        # Create tasks for parallel execution
        tasks = [make_request(i+1) for i in range(num_requests)]
        
        # Execute all requests in parallel
        print("\nExecuting requests...")
        results = await asyncio.gather(*tasks)
        
        # Show results
        print("\nResults:")
        for result in results:
            print(f"  {result}")
        
        # Show final status
        print("\nFinal pool state:")
        status = self.manager.get_status()
        for provider, pool in status.items():
            print(f"  {provider}: {pool['available_clients']} available, {pool['in_use_clients']} in use")
            print(f"  Total requests completed: {pool['metrics']['total_requests']}")
        
        # Show which clients handled requests
        print("\nClient request distribution:")
        pool = self.manager._pools[LLMProvider.ANTHROPIC]
        for i, client in enumerate(pool._clients, 1):
            print(f"  Client {i}: {client._status.total_requests} requests handled")
    
    async def run(self):
        """Run the interactive session."""
        await self.initialize()
        
        print("\n" + "="*60)
        print("🚀 INTERACTIVE LLM SESSION STARTED")
        print("="*60)
        print(f"Provider: Anthropic")
        print(f"Model: claude-3-5-haiku-20241022")
        print(f"System message: {self.system_message}")
        print("\nType /help for commands or start chatting!")
        print("="*60 + "\n")
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("\n👤 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        command_parts = user_input.split(maxsplit=1)
                        command = command_parts[0].lower()
                        args = command_parts[1] if len(command_parts) > 1 else ""
                        
                        if command in ['/exit', '/quit']:
                            print("\n👋 Goodbye!")
                            break
                        elif command == '/help':
                            self.show_help()
                        elif command == '/status':
                            self.show_pool_status()
                        elif command == '/clear':
                            self.history = []
                            print("✓ Conversation history cleared")
                        elif command == '/history':
                            print("\n--- Conversation History ---")
                            for msg in self.history:
                                role = "👤 User" if msg.role == MessageRole.USER else "🤖 Assistant"
                                print(f"{role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                        elif command == '/system':
                            if args:
                                self.system_message = args
                                print(f"✓ System message updated: {self.system_message}")
                            else:
                                print(f"Current system message: {self.system_message}")
                        elif command == '/stream':
                            if args:
                                await self.chat(args, stream=True)
                            else:
                                print("Usage: /stream <your message>")
                        elif command == '/tokens':
                            if args:
                                await self.count_tokens(args)
                            else:
                                print("Usage: /tokens <text to count>")
                        elif command == '/model':
                            print(f"\nModel Information:")
                            print(f"  Provider: {self.client.provider}")
                            print(f"  Model: {self.client.model}")
                            print(f"  Pool size: {len(self.manager._pools[self.client.provider]._clients)}")
                        elif command == '/metrics':
                            await self.show_metrics()
                        elif command == '/clients':
                            self.show_pool_status()
                        elif command == '/parallel':
                            if args and args.isdigit():
                                num = int(args)
                                if 1 <= num <= 10:
                                    await self.test_parallel_requests(num)
                                else:
                                    print("Please specify a number between 1 and 10")
                            else:
                                print("Usage: /parallel <number>")
                                print("Example: /parallel 5")
                        else:
                            print(f"Unknown command: {command}")
                            print("Type /help for available commands")
                    else:
                        # Regular chat
                        await self.chat(user_input)
                        
                except KeyboardInterrupt:
                    print("\n\n⚠️  Use /exit or /quit to leave the session properly")
                    continue
                except Exception as e:
                    logger.error(f"Error in interactive loop: {e}")
                    print(f"\n❌ Error: {e}")
                    
        finally:
            # Cleanup
            if self.manager:
                logger.info("Closing LLM Manager...")
                await self.manager.close()
                logger.info("✓ Manager closed successfully")


async def main():
    """Main entry point."""
    session = InteractiveLLMSession()
    await session.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
        sys.exit(0)