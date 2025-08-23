#!/usr/bin/env python3
"""Performance comparison: With vs Without Instructor"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm import get_llm_manager, LLMProvider, Message, MessageRole


class AnalysisResult(BaseModel):
    """Structured analysis result"""
    main_topic: str = Field(description="Main topic of the text")
    key_points: List[str] = Field(description="3-5 key points")
    sentiment: str = Field(description="positive, negative, or neutral")


async def test_performance():
    """Compare performance with and without Instructor"""
    
    manager = await get_llm_manager()
    client = manager.get_client(LLMProvider.ANTHROPIC)
    
    test_text = """
    The new electric vehicle market is growing rapidly. 
    Tesla continues to lead but faces increasing competition from traditional automakers.
    Battery technology improvements are making EVs more practical for everyday use.
    Government incentives are accelerating adoption rates globally.
    """
    
    messages = [
        Message(MessageRole.USER, f"Analyze this text: {test_text}")
    ]
    
    print("="*60)
    print("PERFORMANCE COMPARISON: Instructor vs Manual Parsing")
    print("="*60)
    
    # Test WITHOUT Instructor (manual parsing needed)
    print("\n1. WITHOUT Instructor (returns string):")
    start = time.time()
    response = await client.complete(messages, max_tokens=200)
    time_without = time.time() - start
    
    print(f"   Time: {time_without:.2f}s")
    print(f"   Type: {type(response).__name__}")
    print(f"   Content length: {len(response.content)} chars")
    print(f"   Would need manual parsing to extract structured data")
    
    # Test WITH Instructor (automatic structured output)
    print("\n2. WITH Instructor (returns structured object):")
    start = time.time()
    result = await client.complete(
        messages, 
        response_model=AnalysisResult,
        max_tokens=200
    )
    time_with = time.time() - start
    
    print(f"   Time: {time_with:.2f}s")
    print(f"   Type: {type(result).__name__}")
    print(f"   Structured Data:")
    print(f"     - Main Topic: {result.main_topic}")
    print(f"     - Sentiment: {result.sentiment}")
    print(f"     - Key Points: {len(result.key_points)} extracted")
    for i, point in enumerate(result.key_points, 1):
        print(f"       {i}. {point[:50]}...")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  • Instructor overhead: {time_with - time_without:.2f}s ({(time_with/time_without - 1)*100:.1f}% slower)")
    print(f"  • But you get guaranteed structured data!")
    print(f"  • No manual parsing needed")
    print(f"  • Type-safe with IDE support")
    print("="*60)


async def main():
    try:
        await test_performance()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())