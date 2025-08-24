#!/usr/bin/env python3
"""
DataMiner Module Usage Examples

This script demonstrates the DataMiner data extraction capabilities:
- Structured data extraction from text
- Multi-stage extraction strategies
- Cognitive extraction with reasoning
- Repository-wide extraction
- Schema validation and confidence scoring
- Batch processing
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "your-key")

from src.dataminer import (
    DataMinerClient,
    ExtractionConfig,
    ProcessingMode,
    DataMinerConfig,
    create_default_config
)
from src.dataminer.models.base import ExtractionSchema
from src.dataminer.models.code import CodeElement, FunctionDefinition, ClassDefinition
from src.dataminer.models.document import DocumentStructure, Section
from src.dataminer.models.repository import RepositoryMap


# Example 1: Basic text extraction with custom schema
class PersonInfo(ExtractionSchema):
    """Schema for extracting person information"""
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: str = Field(description="Person's occupation")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    experience_years: Optional[int] = Field(None, description="Years of experience")
    
    def get_confidence_fields(self) -> List[str]:
        return ["skills", "experience_years"]
    
    def get_required_fields(self) -> List[str]:
        return ["name", "occupation"]


async def basic_extraction_example():
    """Extract structured data from unstructured text"""
    print("\n=== Basic Text Extraction Example ===")
    
    # Initialize DataMiner client
    client = DataMinerClient()
    await client.initialize()
    
    # Sample text
    text = """
    John Smith is a senior software engineer with 8 years of experience.
    He's 32 years old and specializes in Python, JavaScript, and cloud computing.
    John has worked on distributed systems and has expertise in Docker and Kubernetes.
    He's passionate about machine learning and has contributed to several open-source projects.
    """
    
    # Extract person information
    result = await client.extract(
        content=text,
        schema=PersonInfo,
        mode=ProcessingMode.FAST
    )
    
    if result.success:
        person = result.data
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
        print(f"Skills: {', '.join(person.skills)}")
        print(f"Experience: {person.experience_years} years")
        print(f"\nConfidence: {result.confidence.overall:.2f}")
        print(f"Extraction time: {result.processing_time_ms:.2f}ms")
    else:
        print(f"Extraction failed: {', '.join(result.errors)}")
    
    await client.cleanup()


# Example 2: Multi-stage extraction for complex data
class ProjectInfo(ExtractionSchema):
    """Complex project information schema"""
    name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    technologies: List[str] = Field(description="Technologies used")
    team_size: Optional[int] = Field(None, description="Number of team members")
    status: str = Field(description="Current project status")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Project milestones")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    
    def get_confidence_fields(self) -> List[str]:
        return ["technologies", "milestones", "risks"]
    
    def get_required_fields(self) -> List[str]:
        return ["name", "description", "status"]


async def multistage_extraction_example():
    """Demonstrate multi-stage extraction strategy"""
    print("\n=== Multi-Stage Extraction Example ===")
    
    # Configure for thorough extraction
    config = DataMinerConfig(
        extraction=ExtractionConfig(
            default_mode=ProcessingMode.THOROUGH,
            min_confidence_threshold=0.7,
            max_gap_tolerance=0.2
        )
    )
    
    client = DataMinerClient(config)
    await client.initialize()
    
    # Complex project documentation
    project_docs = """
    Project: DataFlow Analytics Platform
    
    Overview:
    DataFlow is a real-time analytics platform designed for processing large-scale 
    streaming data. The project started in January 2024 and is currently in beta testing.
    
    Technical Stack:
    - Backend: Python (FastAPI), Go for performance-critical components
    - Frontend: React with TypeScript
    - Database: PostgreSQL for metadata, ClickHouse for analytics
    - Message Queue: Apache Kafka
    - Deployment: Kubernetes on AWS EKS
    
    Team:
    We have 12 developers, 3 DevOps engineers, 2 data scientists, and 1 project manager.
    
    Milestones:
    - Q1 2024: Architecture design and POC (Completed)
    - Q2 2024: Core platform development (Completed)
    - Q3 2024: Beta testing and performance optimization (In Progress)
    - Q4 2024: Production release and scaling
    
    Current Challenges:
    - Data ingestion bottlenecks at peak loads
    - Complex debugging of distributed transactions
    - Need for better monitoring and alerting
    
    Risk Assessment:
    - Scalability concerns with current architecture
    - Dependency on third-party APIs
    - Potential data privacy compliance issues in EU markets
    """
    
    # Extract with multi-stage strategy
    result = await client.extract(
        content=project_docs,
        schema=ProjectInfo,
        mode=ProcessingMode.THOROUGH
    )
    
    if result.success:
        project = result.data
        print(f"Project: {project.name}")
        print(f"Status: {project.status}")
        print(f"Team Size: {project.team_size}")
        print(f"Technologies: {', '.join(project.technologies[:5])}...")
        print(f"Milestones: {len(project.milestones)}")
        print(f"Identified Risks: {len(project.risks)}")
        
        print(f"\nExtraction Quality:")
        print(f"  Overall Confidence: {result.confidence.overall:.2f}")
        print(f"  Completeness: {result.confidence.completeness:.2f}")
        print(f"  Stages Completed: {', '.join([s.value for s in result.stages_completed])}")
        
        if result.gap_analysis.has_gaps():
            print(f"\nGaps Identified:")
            print(f"  Missing Fields: {', '.join(result.gap_analysis.missing_fields)}")
            print(f"  Recommendations: {result.gap_analysis.recommended_actions[0]}")
    
    await client.cleanup()


# Example 3: Cognitive extraction with reasoning
async def cognitive_extraction_example():
    """Use cognitive reasoning for complex extraction"""
    print("\n=== Cognitive Extraction Example ===")
    
    # Configure for cognitive mode
    config = DataMinerConfig(
        extraction=ExtractionConfig(
            default_mode=ProcessingMode.COGNITIVE,
            use_copilot_reasoning=True,
            copilot_confidence_boost=0.15
        )
    )
    
    client = DataMinerClient(config)
    await client.initialize()
    
    # Complex technical specification
    spec_text = """
    The system shall implement a distributed cache with the following requirements:
    
    1. Support for at least 1 million concurrent connections
    2. Sub-millisecond latency for 99% of read operations
    3. Automatic failover with zero downtime
    4. Data replication across minimum 3 nodes
    5. Support for LRU, LFU, and FIFO eviction policies
    6. TTL support with millisecond precision
    7. Atomic operations for increment/decrement
    8. Pub/sub messaging capabilities
    
    The cache should handle 100,000 requests per second per node and scale
    horizontally. Memory usage should not exceed 75% of available RAM.
    Data persistence is optional but recommended for critical namespaces.
    """
    
    class SystemRequirements(ExtractionSchema):
        """System requirements schema"""
        component_type: str = Field(description="Type of system component")
        functional_requirements: List[str] = Field(description="Functional requirements")
        performance_metrics: Dict[str, Any] = Field(description="Performance requirements")
        scalability_requirements: List[str] = Field(description="Scalability needs")
        constraints: List[str] = Field(description="System constraints")
        
        def get_confidence_fields(self) -> List[str]:
            return ["performance_metrics", "scalability_requirements"]
        
        def get_required_fields(self) -> List[str]:
            return ["component_type", "functional_requirements"]
    
    # Extract with cognitive reasoning
    result = await client.extract(
        content=spec_text,
        schema=SystemRequirements,
        mode=ProcessingMode.COGNITIVE
    )
    
    if result.success:
        reqs = result.data
        print(f"Component: {reqs.component_type}")
        print(f"Functional Requirements: {len(reqs.functional_requirements)}")
        for req in reqs.functional_requirements[:3]:
            print(f"  - {req}")
        
        print(f"\nPerformance Metrics:")
        for key, value in list(reqs.performance_metrics.items())[:3]:
            print(f"  {key}: {value}")
        
        print(f"\nCognitive Analysis:")
        print(f"  Confidence: {result.confidence.overall:.2f}")
        print(f"  Reasoning Quality: {result.confidence.extraction_quality:.2f}")
        print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
    
    await client.cleanup()


# Example 4: Repository-wide code extraction
async def repository_extraction_example():
    """Extract code structure from entire repository"""
    print("\n=== Repository Code Extraction Example ===")
    
    client = DataMinerClient()
    await client.initialize()
    
    # Extract repository structure
    repo_path = Path.cwd() / "src" / "dataminer"
    
    result = await client.extract_from_repository(
        repository_path=repo_path,
        schema=RepositoryMap,
        file_patterns=["*.py"],
        max_files=10,
        mode=ProcessingMode.FAST
    )
    
    if result.success:
        repo_map = result.data
        print(f"Repository: {repo_map.name}")
        print(f"Description: {repo_map.description[:100]}...")
        print(f"Total Files: {len(repo_map.file_map)}")
        print(f"Languages: {', '.join(repo_map.languages)}")
        
        print(f"\nKey Modules:")
        for module in repo_map.modules[:5]:
            print(f"  - {module.name}: {module.description[:50]}...")
        
        print(f"\nDependencies: {', '.join(repo_map.dependencies[:5])}...")
        
        print(f"\nExtraction Metrics:")
        print(f"  Sources Processed: {len(result.sources_processed)}")
        print(f"  Confidence: {result.confidence.overall:.2f}")
        print(f"  Time: {result.processing_time_ms:.2f}ms")
    
    await client.cleanup()


# Example 5: Batch extraction with different schemas
async def batch_extraction_example():
    """Process multiple documents with different schemas"""
    print("\n=== Batch Extraction Example ===")
    
    client = DataMinerClient()
    await client.initialize()
    
    # Different document types
    documents = [
        ("Meeting notes from yesterday's standup...", PersonInfo),
        ("Project specification document...", ProjectInfo),
        ("System architecture overview...", SystemRequirements),
    ]
    
    # Batch process
    results = await client.batch_extract(
        requests=documents,
        mode=ProcessingMode.FAST,
        max_concurrent=3
    )
    
    print(f"Batch Results:")
    print(f"Total Processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.success)}")
    
    for i, result in enumerate(results):
        doc_type = documents[i][1].__name__
        status = "✓" if result.success else "✗"
        confidence = result.confidence.overall if result.success else 0
        print(f"  {status} {doc_type}: confidence={confidence:.2f}")
    
    await client.cleanup()


# Example 6: Custom extraction strategy
async def custom_strategy_example():
    """Use custom extraction configuration"""
    print("\n=== Custom Strategy Example ===")
    
    # Custom configuration
    config = DataMinerConfig(
        extraction=ExtractionConfig(
            default_mode=ProcessingMode.HYBRID,
            min_confidence_threshold=0.8,
            max_gap_tolerance=0.1,
            chunk_size=2000,
            overlap_size=100,
            max_concurrent_extractions=5,
            preferred_provider="anthropic",
            model_settings={
                "temperature": 0.3,
                "max_tokens": 2000
            }
        ),
        enable_result_cache=True,
        cache_directory=Path.home() / ".dataminer_cache"
    )
    
    client = DataMinerClient(config)
    await client.initialize()
    
    # Extract with custom settings
    text = "Your complex document here..."
    
    result = await client.extract(
        content=text,
        schema=DocumentStructure,
        config_overrides={
            "temperature": 0.1,  # Override for this extraction
            "max_iterations": 5
        }
    )
    
    print(f"Custom Extraction Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence.overall:.2f}")
    print(f"  Iterations: {result.iterations_used}")
    
    # Check capabilities
    capabilities = await client.get_extraction_capabilities()
    print(f"\nClient Capabilities:")
    print(f"  Strategies: {', '.join(capabilities['strategies'])}")
    print(f"  Cache Hit Rate: {capabilities['statistics']['cache_hit_rate']:.2f}")
    
    await client.cleanup()


# Example 7: Schema validation and quality checks
async def schema_validation_example():
    """Validate extraction schemas and results"""
    print("\n=== Schema Validation Example ===")
    
    client = DataMinerClient()
    await client.initialize()
    
    # Define a schema to validate
    class ProductReview(ExtractionSchema):
        """Product review schema"""
        product_name: str = Field(description="Product name")
        rating: int = Field(ge=1, le=5, description="Rating 1-5")
        review_text: str = Field(description="Review content")
        pros: List[str] = Field(default_factory=list)
        cons: List[str] = Field(default_factory=list)
        
        def get_confidence_fields(self) -> List[str]:
            return ["pros", "cons"]
        
        def get_required_fields(self) -> List[str]:
            return ["product_name", "rating", "review_text"]
    
    # Validate schema
    validation_result = await client.validate_schema(ProductReview)
    
    print(f"Schema Validation:")
    print(f"  Valid: {validation_result['is_valid']}")
    print(f"  Completeness: {validation_result['completeness_score']:.2f}")
    
    if validation_result.get('warnings'):
        print(f"  Warnings:")
        for warning in validation_result['warnings']:
            print(f"    - {warning}")
    
    # Extract and validate result
    review_text = """
    The new smartphone is amazing! Great camera, long battery life.
    However, it's expensive and a bit heavy. Overall, I rate it 4 out of 5.
    """
    
    result = await client.extract(
        content=review_text,
        schema=ProductReview
    )
    
    if result.success:
        print(f"\nExtraction Quality:")
        print(f"  Schema Compliance: {result.confidence.schema_compliance:.2f}")
        print(f"  Completeness: {result.confidence.completeness:.2f}")
        print(f"  Field Confidence:")
        for field, conf in list(result.confidence.field_confidence.items())[:3]:
            print(f"    {field}: {conf:.2f}")
    
    await client.cleanup()


# Example 8: Progress tracking and callbacks
async def progress_tracking_example():
    """Track extraction progress with callbacks"""
    print("\n=== Progress Tracking Example ===")
    
    client = DataMinerClient()
    await client.initialize()
    
    # Progress callback
    class ProgressTracker:
        def __init__(self):
            self.stages = []
        
        async def on_stage_started(self, stage, progress):
            print(f"  Starting: {stage.value}")
            self.stages.append(stage)
        
        async def on_stage_completed(self, stage, progress):
            print(f"  Completed: {stage.value} ({progress.duration_ms:.2f}ms)")
        
        async def on_stage_progress(self, stage, progress):
            print(f"  Progress: {stage.value} - {progress.progress_percentage:.0f}%")
    
    tracker = ProgressTracker()
    
    # Long document for multi-stage processing
    long_document = """
    [Imagine this is a very long technical document with multiple sections,
    complex data structures, and various types of information that would
    benefit from multi-stage extraction with progress tracking...]
    """ * 10
    
    print("Extraction Progress:")
    
    result = await client.extract(
        content=long_document,
        schema=DocumentStructure,
        mode=ProcessingMode.THOROUGH,
        progress_callback=tracker
    )
    
    print(f"\nCompleted Stages: {len(tracker.stages)}")
    print(f"Total Time: {result.processing_time_ms:.2f}ms")
    print(f"Final Confidence: {result.confidence.overall:.2f}")
    
    await client.cleanup()


async def main():
    """Run all DataMiner examples"""
    print("=" * 60)
    print("DATAMINER MODULE USAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    await basic_extraction_example()
    await multistage_extraction_example()
    await cognitive_extraction_example()
    await repository_extraction_example()
    await batch_extraction_example()
    await custom_strategy_example()
    await schema_validation_example()
    await progress_tracking_example()
    
    print("\n" + "=" * 60)
    print("All DataMiner examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())