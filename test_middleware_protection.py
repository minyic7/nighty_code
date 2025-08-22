#!/usr/bin/env python3
"""
Enhanced middleware rate limiting protection test with better error handling.
Diagnoses issues and provides detailed insights into performance problems.
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import statistics
import traceback
from dataclasses import dataclass
from collections import defaultdict, Counter

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
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f"enhanced_middleware_test_{TIMESTAMP}.log"

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise but keep important middleware logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)


@dataclass
class TestResult:
    """Enhanced test result tracking"""
    id: int
    success: bool
    duration: float
    start_time: float
    end_time: float
    client_id: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    is_rate_limit: bool = False
    is_timeout: bool = False
    is_auth_error: bool = False
    is_connection_error: bool = False
    retry_count: int = 0
    response_content: Optional[str] = None
    tokens_used: Optional[int] = None


class EnhancedMiddlewareProtectionTester:
    """Enhanced middleware protection tester with better diagnostics"""

    def __init__(self):
        self.manager = None
        self.client = None
        self.results: List[TestResult] = []
        self.error_categories = defaultdict(int)
        self.client_performance = defaultdict(list)

    async def initialize(self):
        """Initialize manager with enhanced error handling"""
        logger.info("=" * 80)
        logger.info("ENHANCED MIDDLEWARE RATE LIMIT PROTECTION TEST")
        logger.info("=" * 80)

        try:
            self.manager = await get_llm_manager()

            # Verify Anthropic pool exists
            if LLMProvider.ANTHROPIC not in self.manager._pools:
                raise ValueError("Anthropic provider not configured")

            pool = self.manager._pools[LLMProvider.ANTHROPIC]

            # Log initial pool state
            logger.info(f"Initial pool state: {len(pool._clients)} clients, {len(pool.configs)} configurations")

            # Pre-initialize all clients with error handling
            num_configs = len(pool.configs)
            logger.info(f"Found {num_configs} API key configurations")

            if num_configs == 0:
                raise ValueError("No Anthropic API configurations found")

            # Initialize clients one by one with error handling
            for i in range(num_configs):
                if len(pool._clients) < num_configs:
                    try:
                        logger.info(f"Creating client {len(pool._clients) + 1}/{num_configs}...")
                        await pool._create_client()
                        logger.info(f"  ✓ Client {len(pool._clients)} created successfully")
                    except Exception as e:
                        logger.error(f"  ✗ Failed to create client {i + 1}: {e}")
                        # Log the full traceback for debugging
                        logger.debug(traceback.format_exc())

            if len(pool._clients) == 0:
                raise ValueError("Failed to create any clients")

            self.client = self.manager.get_client(LLMProvider.ANTHROPIC)

            # Test a simple request to verify functionality
            logger.info("Testing basic connectivity...")
            try:
                test_messages = [Message(MessageRole.USER, "Say 'test'")]
                test_response = await asyncio.wait_for(
                    self.client.complete(test_messages, max_tokens=5),
                    timeout=30.0
                )
                logger.info(f"✓ Basic connectivity test passed: {test_response.content[:50]}")
            except Exception as e:
                logger.error(f"✗ Basic connectivity test failed: {e}")
                logger.debug(traceback.format_exc())
                raise

            # Log enhanced configuration
            logger.info(f"\n✓ Pool initialized with {len(pool._clients)} active clients")

            total_capacity = 0
            for i, client_wrapper in enumerate(pool._clients, 1):
                rate_limiter = getattr(client_wrapper, 'rate_limiter', None)
                if rate_limiter and hasattr(rate_limiter, 'rate_limiter'):
                    config = rate_limiter.rate_limiter.config
                    logger.info(
                        f"  Client {i}: {config.requests_per_minute} req/min, "
                        f"{config.input_tokens_per_minute} input TPM, "
                        f"{config.output_tokens_per_minute} output TPM"
                    )
                    total_capacity += config.requests_per_minute
                else:
                    logger.warning(f"  Client {i}: No rate limiter configured")

            logger.info(f"\nTotal theoretical capacity: {total_capacity} requests/minute")
            logger.info("Middleware should prevent ALL 429 errors")

            # Log tokenizer information
            if hasattr(pool._clients[0], 'rate_limiter'):
                limiter = pool._clients[0].rate_limiter
                if hasattr(limiter, 'get_tokenizer_info'):
                    tokenizer_info = limiter.get_tokenizer_info()
                    logger.info(f"Tokenizer: {tokenizer_info}")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            logger.debug(traceback.format_exc())
            raise

    async def make_request_with_retry(self, request_id: int, complexity: str = "simple",
                                      max_retries: int = 3) -> TestResult:
        """Make a request with enhanced error handling and optional retry"""
        start_time = time.time()

        # Create message based on complexity
        if complexity == "simple":
            content = f"Reply with just the number '{request_id}'"
            max_tokens = 10
        elif complexity == "medium":
            content = f"Count from 1 to {min(5, request_id % 5 + 1)} and add the word 'done'"
            max_tokens = 30
        else:  # complex
            content = f"Write a {min(3, request_id % 3 + 1)} sentence summary of what counting means"
            max_tokens = 100

        messages = [Message(MessageRole.USER, content)]

        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Request {request_id}, attempt {attempt + 1}")

                # Set timeout based on complexity
                timeout = 30.0 if complexity == "simple" else 45.0

                response = await asyncio.wait_for(
                    self.client.complete(messages, max_tokens=max_tokens),
                    timeout=timeout
                )

                end_time = time.time()
                duration = end_time - start_time

                # Extract client performance data
                client_id = getattr(response, 'client_id', 'unknown')
                self.client_performance[client_id].append(duration)

                result = TestResult(
                    id=request_id,
                    success=True,
                    duration=duration,
                    start_time=start_time,
                    end_time=end_time,
                    client_id=client_id,
                    response_content=response.content[:100],
                    tokens_used=getattr(response, 'usage', {}).get('total_tokens', None),
                    retry_count=attempt
                )

                logger.debug(f"Request {request_id} succeeded in {duration:.2f}s via {client_id}")
                return result

            except asyncio.TimeoutError:
                error_msg = f"Request timeout after {timeout}s"
                self.error_categories['timeout'] += 1
                logger.warning(f"Request {request_id} timeout on attempt {attempt + 1}")

                if attempt == max_retries:
                    return TestResult(
                        id=request_id,
                        success=False,
                        duration=time.time() - start_time,
                        start_time=start_time,
                        end_time=time.time(),
                        error=error_msg,
                        error_type="TimeoutError",
                        is_timeout=True,
                        retry_count=attempt
                    )

                # Wait before retry
                await asyncio.sleep(min(2 ** attempt, 5))

            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__

                # Categorize error
                is_rate_limit = '429' in error_str or 'rate_limit' in error_str.lower()
                is_auth_error = '401' in error_str or 'unauthorized' in error_str.lower() or 'api_key' in error_str.lower()
                is_connection_error = 'connection' in error_str.lower() or 'network' in error_str.lower()

                if is_rate_limit:
                    self.error_categories['rate_limit'] += 1
                    logger.error(f"CRITICAL: Request {request_id} hit rate limit: {error_str}")
                elif is_auth_error:
                    self.error_categories['auth'] += 1
                    logger.error(f"Auth error on request {request_id}: {error_str}")
                elif is_connection_error:
                    self.error_categories['connection'] += 1
                    logger.warning(f"Connection error on request {request_id}: {error_str}")
                else:
                    self.error_categories['other'] += 1
                    logger.warning(f"Other error on request {request_id}: {error_str}")

                # Don't retry rate limit or auth errors
                if is_rate_limit or is_auth_error or attempt == max_retries:
                    return TestResult(
                        id=request_id,
                        success=False,
                        duration=time.time() - start_time,
                        start_time=start_time,
                        end_time=time.time(),
                        error=error_str[:200],
                        error_type=error_type,
                        is_rate_limit=is_rate_limit,
                        is_auth_error=is_auth_error,
                        is_connection_error=is_connection_error,
                        retry_count=attempt
                    )

                # Wait before retry for retryable errors
                await asyncio.sleep(min(2 ** attempt, 5))

        # Should never reach here
        return TestResult(
            id=request_id,
            success=False,
            duration=time.time() - start_time,
            start_time=start_time,
            end_time=time.time(),
            error="Max retries exceeded",
            error_type="MaxRetriesError",
            retry_count=max_retries
        )

    async def test_basic_functionality(self, num_requests: int = 10):
        """Test basic functionality with small load"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST 0: BASIC FUNCTIONALITY - {num_requests} sequential requests")
        logger.info(f"{'=' * 60}")

        results = []
        for i in range(1, num_requests + 1):
            logger.info(f"Making request {i}/{num_requests}...")
            result = await self.make_request_with_retry(i, "simple")
            results.append(result)

            if not result.success:
                logger.error(f"Request {i} failed: {result.error}")
            else:
                logger.info(f"Request {i} succeeded in {result.duration:.2f}s")

            # Small delay between requests
            await asyncio.sleep(0.5)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        logger.info(f"\nBasic functionality results:")
        logger.info(f"  ✓ Successful: {len(successful)}/{num_requests}")
        logger.info(f"  ✗ Failed: {len(failed)}")

        if len(successful) < num_requests * 0.8:  # Less than 80% success
            logger.error("Basic functionality test failed - too many errors")
            logger.error("This indicates fundamental configuration issues")

            # Analyze errors
            error_types = Counter(r.error_type for r in failed if r.error_type)
            logger.error(f"Error breakdown: {dict(error_types)}")

            return False

        self.results.extend(results)
        return True

    async def test_burst_protection(self, num_requests: int = 50):
        """Test burst protection with better analysis"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST 1: BURST PROTECTION - {num_requests} concurrent requests")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # Launch concurrent requests
        tasks = [
            self.make_request_with_retry(i, "simple", max_retries=1)
            for i in range(1, num_requests + 1)
        ]
        results = await asyncio.gather(*tasks)

        duration = time.time() - start_time

        # Enhanced analysis
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        rate_limited = [r for r in results if r.is_rate_limit]

        logger.info(f"\nCompleted in {duration:.2f}s")
        logger.info(f"Results:")
        logger.info(f"  ✓ Successful: {len(successful)}/{num_requests} ({len(successful) / num_requests * 100:.1f}%)")
        logger.info(f"  ✗ Failed: {len(failed)}/{num_requests}")
        logger.info(f"  🚫 Rate limit errors: {len(rate_limited)}")

        # Error analysis
        if failed:
            error_breakdown = Counter(r.error_type for r in failed if r.error_type)
            logger.info(f"  Error breakdown: {dict(error_breakdown)}")

            # Show sample errors
            for error_type, count in error_breakdown.most_common(3):
                sample = next(r for r in failed if r.error_type == error_type)
                logger.info(f"    {error_type} example: {sample.error[:100]}")

        # Performance analysis for successful requests
        if successful:
            durations = [r.duration for r in successful]
            logger.info(f"\nPerformance metrics:")
            logger.info(f"  Min response time: {min(durations):.2f}s")
            logger.info(f"  Max response time: {max(durations):.2f}s")
            logger.info(f"  Average: {statistics.mean(durations):.2f}s")
            logger.info(f"  Median: {statistics.median(durations):.2f}s")
            logger.info(f"  95th percentile: {sorted(durations)[int(len(durations) * 0.95)]:.2f}s")

            # Client distribution
            client_dist = Counter(r.client_id for r in successful if r.client_id)
            logger.info(f"\nClient distribution:")
            for client_id, count in client_dist.items():
                logger.info(f"  {client_id}: {count} requests ({count / len(successful) * 100:.1f}%)")

        # Rate limit assessment
        if rate_limited:
            logger.error(f"\n❌ MIDDLEWARE FAILURE: {len(rate_limited)} requests hit rate limits!")
            for rl in rate_limited[:3]:
                logger.error(f"  Request {rl.id}: {rl.error}")
        else:
            logger.info(f"\n✅ SUCCESS: Middleware prevented all rate limit errors!")

        self.results.extend(results)
        return len(rate_limited) == 0

    async def test_sustained_load(self, duration_seconds: int = 30, target_rps: float = 0.5):
        """Test sustained load with better pacing"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST 2: SUSTAINED LOAD - {duration_seconds}s at {target_rps:.1f} req/s")
        logger.info(f"{'=' * 60}")

        expected_total = int(duration_seconds * target_rps)
        logger.info(f"Expected total requests: {expected_total}")

        start_time = time.time()
        results = []
        request_id = 1

        # More precise timing
        next_request_time = start_time

        while time.time() - start_time < duration_seconds:
            current_time = time.time()

            if current_time >= next_request_time:
                # Launch request
                result = await self.make_request_with_retry(request_id, "simple", max_retries=0)
                results.append(result)

                # Log progress
                if request_id % 5 == 0:
                    elapsed = current_time - start_time
                    rate = len(results) / elapsed * 60  # req/min
                    logger.info(f"Progress: {request_id} requests in {elapsed:.1f}s ({rate:.1f} req/min)")

                request_id += 1
                next_request_time += 1.0 / target_rps
            else:
                # Wait until next request time
                await asyncio.sleep(min(0.1, next_request_time - current_time))

        total_duration = time.time() - start_time

        # Analysis
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        rate_limited = [r for r in results if r.is_rate_limit]

        actual_rate = len(results) / (total_duration / 60)  # req/min
        success_rate = len(successful) / (total_duration / 60) if successful else 0

        logger.info(f"\nSustained load results:")
        logger.info(f"  Duration: {total_duration:.2f}s")
        logger.info(f"  Total requests: {len(results)} (expected: {expected_total})")
        logger.info(f"  ✓ Successful: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
        logger.info(f"  ✗ Failed: {len(failed)}")
        logger.info(f"  🚫 Rate limit errors: {len(rate_limited)}")
        logger.info(f"  Actual rate: {actual_rate:.1f} req/min")
        logger.info(f"  Success rate: {success_rate:.1f} req/min")

        # Error analysis
        if failed:
            error_breakdown = Counter(r.error_type for r in failed if r.error_type)
            logger.info(f"  Error types: {dict(error_breakdown)}")

        if rate_limited:
            logger.error(f"\n❌ SUSTAINED LOAD FAILURE: Rate limits hit during sustained load!")
        else:
            logger.info(f"\n✅ SUCCESS: No rate limit errors during sustained load!")

        self.results.extend(results)
        return len(rate_limited) == 0

    def generate_enhanced_summary(self):
        """Generate comprehensive test summary with diagnostics"""
        logger.info("\n" + "=" * 80)
        logger.info("ENHANCED MIDDLEWARE TEST SUMMARY")
        logger.info("=" * 80)

        if not self.results:
            logger.error("No test results to analyze!")
            return

        # Basic statistics
        total_requests = len(self.results)
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        rate_limited = [r for r in self.results if r.is_rate_limit]

        logger.info(f"\nOverall Results:")
        logger.info(f"  Total requests: {total_requests}")
        logger.info(f"  ✓ Successful: {len(successful)} ({len(successful) / total_requests * 100:.1f}%)")
        logger.info(f"  ✗ Failed: {len(failed)} ({len(failed) / total_requests * 100:.1f}%)")
        logger.info(f"  🚫 Rate limit errors: {len(rate_limited)}")

        # Error analysis
        logger.info(f"\nError Analysis:")
        for category, count in self.error_categories.items():
            percentage = count / total_requests * 100
            logger.info(f"  {category.title()}: {count} ({percentage:.1f}%)")

        # Performance analysis
        if successful:
            durations = [r.duration for r in successful]
            logger.info(f"\nPerformance Analysis:")
            logger.info(f"  Response times - Min: {min(durations):.2f}s, Max: {max(durations):.2f}s")
            logger.info(f"  Average: {statistics.mean(durations):.2f}s, Median: {statistics.median(durations):.2f}s")

            if len(durations) > 1:
                logger.info(f"  Std deviation: {statistics.stdev(durations):.2f}s")

        # Client performance analysis
        if self.client_performance:
            logger.info(f"\nClient Performance:")
            for client_id, durations in self.client_performance.items():
                if durations:
                    avg_duration = statistics.mean(durations)
                    logger.info(f"  {client_id}: {len(durations)} requests, avg {avg_duration:.2f}s")

        # Rate limit protection assessment
        logger.info(f"\nRate Limit Protection Assessment:")
        if len(rate_limited) == 0:
            logger.info(f"  🎉 PERFECT: Middleware prevented all rate limit errors!")
            logger.info(f"  ✓ Token bucket algorithm working correctly")
            logger.info(f"  ✓ Request queueing functioning properly")
        else:
            logger.error(f"  ❌ FAILED: {len(rate_limited)} requests hit rate limits")
            logger.error(f"  ✗ Middleware configuration needs adjustment")

        # Recommendations
        logger.info(f"\nRecommendations:")

        if len(failed) > total_requests * 0.5:
            logger.warning(f"  ⚠️  High failure rate ({len(failed) / total_requests * 100:.1f}%) indicates:")
            if self.error_categories['timeout'] > 0:
                logger.warning(f"    - Consider increasing request timeouts")
            if self.error_categories['auth'] > 0:
                logger.warning(f"    - Check API key configuration")
            if self.error_categories['connection'] > 0:
                logger.warning(f"    - Check network connectivity")

        if len(successful) < total_requests * 0.2:
            logger.error(f"  🚨 Very low success rate - fundamental issues present")
            logger.error(f"    - Verify API keys are valid and have quota")
            logger.error(f"    - Check network connectivity")
            logger.error(f"    - Review rate limiter configuration")

        if successful and statistics.mean([r.duration for r in successful]) > 10:
            logger.warning(f"  ⏰ High average response times suggest:")
            logger.warning(f"    - Rate limiting may be too aggressive")
            logger.warning(f"    - Consider adjusting token bucket sizes")

        # Save detailed results
        results_file = LOG_DIR / f"enhanced_test_results_{TIMESTAMP}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_requests': total_requests,
                    'successful': len(successful),
                    'failed': len(failed),
                    'rate_limit_errors': len(rate_limited),
                    'error_categories': dict(self.error_categories),
                    'protection_success': len(rate_limited) == 0
                },
                'client_performance': {
                    client_id: {
                        'request_count': len(durations),
                        'avg_duration': statistics.mean(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations)
                    }
                    for client_id, durations in self.client_performance.items()
                    if durations
                },
                'sample_results': [
                    {
                        'id': r.id,
                        'success': r.success,
                        'duration': r.duration,
                        'error_type': r.error_type,
                        'client_id': r.client_id
                    }
                    for r in self.results[:50]  # First 50 results
                ]
            }, f, indent=2, default=str)

        logger.info(f"\n📊 Detailed results: {results_file}")
        logger.info(f"📜 Full log: {LOG_FILE}")

    async def cleanup(self):
        """Clean up resources"""
        if self.manager:
            try:
                await self.manager.close()
                logger.info("Manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing manager: {e}")


async def main():
    """Run enhanced middleware protection tests"""
    tester = EnhancedMiddlewareProtectionTester()

    try:
        await tester.initialize()

        # Test 0: Basic functionality
        logger.info("Starting basic functionality test...")
        basic_success = await tester.test_basic_functionality(10)

        if not basic_success:
            logger.error("Basic functionality test failed - stopping further tests")
            tester.generate_enhanced_summary()
            return

        await asyncio.sleep(3)  # Cool down

        # Test 1: Burst protection (reduced size for debugging)
        logger.info("Starting burst protection test...")
        burst_success = await tester.test_burst_protection(50)
        await asyncio.sleep(3)  # Cool down

        # Test 2: Sustained load (reduced duration for debugging)
        logger.info("Starting sustained load test...")
        sustained_success = await tester.test_sustained_load(30, 0.5)

        # Generate comprehensive summary
        tester.generate_enhanced_summary()

        # Final assessment
        all_passed = basic_success and burst_success and sustained_success
        if all_passed:
            logger.info("\n✅ All tests PASSED! Middleware provides complete protection.")
        else:
            logger.warning("\n⚠️  Some tests had issues. Review the detailed analysis above.")

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        logger.debug(traceback.format_exc())
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())