# DataMiner Module

A powerful data extraction system that uses LLMs to extract structured data from unstructured content with multiple strategies, validation, and confidence scoring.

## Features

### 📊 Extraction Strategies
- **Simple Extraction**: Fast single-pass extraction for straightforward data
- **Multi-Stage Extraction**: Discovery → Initial → Refinement → Validation pipeline
- **Cognitive Extraction**: Leverages reasoning for complex understanding
- **Reflective Extraction**: Iterative improvement based on quality analysis
- **Hybrid Mode**: Combines strategies for optimal results

### 🎯 Structured Output
- **Pydantic Schema Support**: Type-safe extraction with validation
- **Custom Schema Definition**: Create domain-specific extraction schemas
- **Nested Structures**: Handle hierarchical and relational data
- **Field-level Confidence**: Track confidence per extracted field

### 🔍 Advanced Capabilities
- **Repository-wide Extraction**: Process entire codebases or document sets
- **Batch Processing**: Concurrent extraction with multiple schemas
- **Gap Analysis**: Identify missing or low-confidence data
- **Schema Validation**: Ensure schema quality before extraction
- **Progress Tracking**: Real-time extraction progress monitoring

### 🛡️ Quality Assurance
- **Confidence Scoring**: Multi-dimensional confidence metrics
- **Validation Stages**: Built-in data validation pipeline
- **Error Recovery**: Graceful handling of extraction failures
- **Result Caching**: Intelligent caching for repeated extractions

## Installation

```bash
# Install required dependencies
pip install pydantic asyncio typing-extensions
```

## Quick Start

```python
import asyncio
from src.dataminer import DataMinerClient, ProcessingMode
from src.dataminer.models.base import ExtractionSchema
from pydantic import Field
from typing import List, Optional

# Define your schema
class PersonInfo(ExtractionSchema):
    name: str = Field(description="Person's full name")
    age: Optional[int] = Field(None, description="Person's age")
    occupation: str = Field(description="Person's job")
    skills: List[str] = Field(default_factory=list, description="Skills")
    
    def get_confidence_fields(self) -> List[str]:
        return ["skills"]
    
    def get_required_fields(self) -> List[str]:
        return ["name", "occupation"]

async def main():
    # Initialize client
    client = DataMinerClient()
    await client.initialize()
    
    # Extract data
    text = "John Smith is a 30-year-old software engineer skilled in Python and Java."
    
    result = await client.extract(
        content=text,
        schema=PersonInfo,
        mode=ProcessingMode.FAST
    )
    
    if result.success:
        print(f"Name: {result.data.name}")
        print(f"Skills: {', '.join(result.data.skills)}")
        print(f"Confidence: {result.confidence.overall:.2f}")
    
    await client.cleanup()

asyncio.run(main())
```

## Processing Modes

### Fast Mode
Best for simple, well-structured content:
```python
result = await client.extract(
    content=text,
    schema=MySchema,
    mode=ProcessingMode.FAST
)
```

### Thorough Mode
Multi-stage extraction for complex content:
```python
result = await client.extract(
    content=complex_document,
    schema=MySchema,
    mode=ProcessingMode.THOROUGH
)
```

### Cognitive Mode
Uses reasoning for ambiguous content:
```python
result = await client.extract(
    content=technical_spec,
    schema=MySchema,
    mode=ProcessingMode.COGNITIVE
)
```

### Hybrid Mode
Combines strategies adaptively:
```python
result = await client.extract(
    content=mixed_content,
    schema=MySchema,
    mode=ProcessingMode.HYBRID
)
```

## Built-in Schema Models

### Code Extraction
```python
from src.dataminer.models.code import CodeElement, FunctionDefinition, ClassDefinition

# Extract code structure
result = await client.extract(
    content=source_code,
    schema=CodeElement
)

# Access extracted functions
for func in result.data.functions:
    print(f"Function: {func.name}")
    print(f"  Parameters: {func.parameters}")
    print(f"  Returns: {func.return_type}")
```

### Document Structure
```python
from src.dataminer.models.document import DocumentStructure, Section

# Extract document structure
result = await client.extract(
    content=document_text,
    schema=DocumentStructure
)

# Navigate sections
for section in result.data.sections:
    print(f"Section: {section.title}")
    print(f"  Level: {section.level}")
    print(f"  Content: {section.content[:100]}...")
```

### Repository Mapping
```python
from src.dataminer.models.repository import RepositoryMap

# Extract repository structure
result = await client.extract_from_repository(
    repository_path="./my_project",
    schema=RepositoryMap,
    file_patterns=["*.py", "*.js"],
    max_files=50
)

print(f"Repository: {result.data.name}")
print(f"Languages: {', '.join(result.data.languages)}")
print(f"Modules: {len(result.data.modules)}")
```

## Repository Extraction

Process entire codebases or document collections:

```python
from pathlib import Path

# Extract from repository
result = await client.extract_from_repository(
    repository_path=Path("./my_project"),
    schema=ProjectDocumentation,
    file_patterns=["*.md", "*.py"],
    exclude_patterns=["test_*", "__pycache__"],
    max_files=100,
    mode=ProcessingMode.THOROUGH
)

print(f"Files processed: {len(result.sources_processed)}")
print(f"Extraction confidence: {result.confidence.overall:.2f}")
```

## Batch Processing

Extract multiple documents with different schemas:

```python
# Prepare batch requests
requests = [
    ("document1.txt", Schema1),
    ("document2.txt", Schema2),
    ("document3.txt", Schema3),
]

# Batch extract
results = await client.batch_extract(
    requests=requests,
    mode=ProcessingMode.FAST,
    max_concurrent=5
)

# Process results
for i, result in enumerate(results):
    if result.success:
        print(f"Document {i+1}: Success (confidence={result.confidence.overall:.2f})")
    else:
        print(f"Document {i+1}: Failed - {result.errors}")
```

## Configuration

```python
from src.dataminer import DataMinerConfig, ExtractionConfig

config = DataMinerConfig(
    extraction=ExtractionConfig(
        # Processing settings
        default_mode=ProcessingMode.THOROUGH,
        min_confidence_threshold=0.7,
        max_gap_tolerance=0.2,
        
        # Chunking settings
        chunk_size=2000,
        overlap_size=200,
        
        # Concurrency
        max_concurrent_extractions=5,
        
        # LLM settings
        preferred_provider="anthropic",
        temperature=0.3,
        max_tokens=2000,
        
        # Caching
        cache_ttl_seconds=3600,
        enable_result_cache=True
    ),
    
    # System settings
    log_level="INFO",
    memory_limit_mb=1024,
    cache_directory=Path.home() / ".dataminer_cache"
)

client = DataMinerClient(config)
```

## Progress Tracking

Monitor extraction progress in real-time:

```python
class ProgressTracker:
    async def on_stage_started(self, stage, progress):
        print(f"Starting: {stage.value}")
    
    async def on_stage_completed(self, stage, progress):
        print(f"Completed: {stage.value} ({progress.duration_ms:.2f}ms)")
    
    async def on_stage_progress(self, stage, progress):
        print(f"Progress: {stage.value} - {progress.progress_percentage:.0f}%")

tracker = ProgressTracker()

result = await client.extract(
    content=long_document,
    schema=MySchema,
    mode=ProcessingMode.THOROUGH,
    progress_callback=tracker
)
```

## Schema Validation

Validate schemas before extraction:

```python
# Validate schema design
validation = await client.validate_schema(MySchema)

print(f"Schema valid: {validation['is_valid']}")
print(f"Completeness: {validation['completeness_score']:.2f}")

if validation.get('warnings'):
    for warning in validation['warnings']:
        print(f"Warning: {warning}")

if validation.get('suggestions'):
    for suggestion in validation['suggestions']:
        print(f"Suggestion: {suggestion}")
```

## Confidence Metrics

Understanding extraction confidence:

```python
result = await client.extract(content=text, schema=MySchema)

if result.success:
    # Overall confidence
    print(f"Overall: {result.confidence.overall:.2f}")
    
    # Component confidence
    print(f"Extraction Quality: {result.confidence.extraction_quality:.2f}")
    print(f"Schema Compliance: {result.confidence.schema_compliance:.2f}")
    print(f"Completeness: {result.confidence.completeness:.2f}")
    
    # Field-level confidence
    for field, confidence in result.confidence.field_confidence.items():
        print(f"  {field}: {confidence:.2f}")
    
    # Gap analysis
    if result.gap_analysis.has_gaps():
        print(f"Missing fields: {result.gap_analysis.missing_fields}")
        print(f"Low confidence: {result.gap_analysis.low_confidence_fields}")
```

## Creating Custom Strategies

```python
from src.dataminer.strategies.base import BaseExtractionStrategy

class CustomStrategy(BaseExtractionStrategy):
    def __init__(self):
        super().__init__("CustomStrategy")
    
    def supports_mode(self, mode):
        return mode == ProcessingMode.CUSTOM
    
    async def extract(self, request, config):
        # Custom extraction logic
        context = ExtractionContext(request=request, config=config)
        
        # Your extraction implementation
        data = await self.custom_extraction_logic(context)
        
        return await self._create_result(context, data)
```

## Error Handling

```python
from src.dataminer.core.exceptions import (
    DataMinerError,
    ExtractionError,
    ValidationError,
    SchemaError
)

try:
    result = await client.extract(content=text, schema=MySchema)
except ValidationError as e:
    print(f"Validation failed: {e}")
except ExtractionError as e:
    print(f"Extraction failed: {e}")
except DataMinerError as e:
    print(f"General error: {e}")
```

## Module Structure

```
src/dataminer/
├── __init__.py           # Main exports
├── client.py             # DataMiner client
├── core/
│   ├── config.py        # Configuration
│   ├── exceptions.py    # Custom exceptions
│   └── types.py         # Type definitions
├── models/
│   ├── base.py          # Base schema models
│   ├── code.py          # Code extraction schemas
│   ├── document.py      # Document schemas
│   └── repository.py    # Repository schemas
├── strategies/
│   ├── base.py          # Base strategy
│   ├── simple.py        # Simple extraction
│   ├── multistage.py    # Multi-stage extraction
│   ├── cognitive.py     # Cognitive extraction
│   └── reflective.py    # Reflective extraction
└── utils/
    ├── confidence.py    # Confidence scoring
    ├── repository.py    # Repository analysis
    └── validator.py     # Schema validation
```

## Best Practices

1. **Schema Design**: Keep schemas focused and well-documented
2. **Mode Selection**: Choose appropriate mode for content complexity
3. **Chunk Size**: Adjust chunk size based on content structure
4. **Caching**: Enable caching for repeated extractions
5. **Validation**: Always validate schemas before production use

## Performance Optimization

- Use FAST mode for simple, well-structured content
- Enable caching for repeated extractions
- Batch process similar documents
- Limit concurrent extractions based on resources
- Use appropriate chunk sizes for your content

## License

MIT License - See LICENSE file for details