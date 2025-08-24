# src/dataminer/core/types.py
"""Core types and data structures for DataMiner"""

from typing import Dict, List, Optional, Any, Union, Literal, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio


T = TypeVar('T', bound=BaseModel)


class ProcessingMode(str, Enum):
    """Data processing modes"""
    FAST = "fast"           # Quick single-pass extraction
    THOROUGH = "thorough"   # Multi-stage with validation
    COGNITIVE = "cognitive" # Uses Copilot cognitive reasoning
    HYBRID = "hybrid"       # Adaptive based on complexity


class ExtractionStage(str, Enum):
    """Stages in multi-stage extraction"""
    DISCOVERY = "discovery"       # Find relevant content
    INITIAL = "initial"           # First pass extraction
    REFINEMENT = "refinement"     # Improve and validate
    VALIDATION = "validation"     # Final validation
    GAP_ANALYSIS = "gap_analysis" # Identify missing data


class ConfidenceLevel(str, Enum):
    """Confidence level classifications"""
    VERY_LOW = "very_low"     # < 0.3
    LOW = "low"               # 0.3 - 0.5
    MEDIUM = "medium"         # 0.5 - 0.7
    HIGH = "high"             # 0.7 - 0.85
    VERY_HIGH = "very_high"   # > 0.85


@dataclass
class ConfidenceMetrics:
    """Detailed confidence metrics for extraction"""
    overall: float = field(default=0.0)
    extraction_quality: float = field(default=0.0)
    schema_compliance: float = field(default=0.0)
    completeness: float = field(default=0.0)
    consistency: float = field(default=0.0)
    
    # Stage-specific confidence
    stage_scores: Dict[ExtractionStage, float] = field(default_factory=dict)
    
    # Detailed breakdown
    field_confidence: Dict[str, float] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    
    def get_level(self) -> ConfidenceLevel:
        """Get confidence level classification"""
        if self.overall < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.overall < 0.5:
            return ConfidenceLevel.LOW
        elif self.overall < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.overall < 0.85:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def update_overall(self):
        """Calculate overall confidence as weighted average"""
        weights = {
            'extraction_quality': 0.3,
            'schema_compliance': 0.25,
            'completeness': 0.25,
            'consistency': 0.2
        }
        
        scores = [
            self.extraction_quality * weights['extraction_quality'],
            self.schema_compliance * weights['schema_compliance'],
            self.completeness * weights['completeness'],
            self.consistency * weights['consistency']
        ]
        
        self.overall = sum(scores)
        return self.overall


@dataclass
class GapAnalysis:
    """Analysis of missing or incomplete data"""
    missing_fields: List[str] = field(default_factory=list)
    incomplete_fields: List[str] = field(default_factory=list)
    low_confidence_fields: List[str] = field(default_factory=list)
    
    # Suggestions for improvement
    recommended_actions: List[str] = field(default_factory=list)
    additional_sources: List[str] = field(default_factory=list)
    
    # Quantitative measures
    completeness_score: float = field(default=0.0)
    coverage_percentage: float = field(default=0.0)
    
    def has_gaps(self) -> bool:
        """Check if there are any identified gaps"""
        return bool(
            self.missing_fields or 
            self.incomplete_fields or 
            self.low_confidence_fields
        )


class ExtractionRequest(BaseModel, Generic[T]):
    """Request for data extraction"""
    # Target schema
    schema_model: type[T] = Field(description="Pydantic model defining the target schema")
    
    # Input data
    content: Union[str, List[str]] = Field(description="Content to extract from")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Processing configuration
    mode: ProcessingMode = Field(default=ProcessingMode.THOROUGH)
    max_iterations: int = Field(default=3, description="Max refinement iterations")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Stage configuration
    enabled_stages: List[ExtractionStage] = Field(
        default_factory=lambda: [
            ExtractionStage.DISCOVERY,
            ExtractionStage.INITIAL,
            ExtractionStage.REFINEMENT,
            ExtractionStage.VALIDATION
        ]
    )
    
    # Repository-specific options
    file_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    max_files: Optional[int] = None
    max_file_size: int = Field(default=10*1024*1024)  # 10MB
    
    # LLM configuration
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: Optional[int] = None
    model_preference: Optional[str] = None
    
    # Metadata
    request_id: str = Field(default_factory=lambda: f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    timestamp: datetime = Field(default_factory=datetime.now)


class ExtractionResult(BaseModel, Generic[T]):
    """Result of data extraction"""
    # Core results
    data: Optional[T] = Field(description="Extracted data conforming to schema")
    raw_extractions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Quality metrics
    confidence: ConfidenceMetrics = Field(default_factory=ConfidenceMetrics)
    gap_analysis: GapAnalysis = Field(default_factory=GapAnalysis)
    
    # Processing information
    stages_completed: List[ExtractionStage] = Field(default_factory=list)
    iterations_used: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)
    
    # Sources and provenance
    sources_processed: List[str] = Field(default_factory=list)
    source_confidence: Dict[str, float] = Field(default_factory=dict)
    extraction_path: List[str] = Field(default_factory=list)
    
    # Error and warning information
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    request_id: str = Field(description="ID of the originating request")
    completed_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if extraction was successful"""
        return self.data is not None and not self.errors
    
    @property
    def needs_refinement(self) -> bool:
        """Check if result would benefit from refinement"""
        return (
            self.confidence.overall < 0.7 or
            self.gap_analysis.has_gaps() or
            bool(self.warnings)
        )


class ExtractionConfig(BaseModel):
    """Configuration for extraction process"""
    # Processing settings
    default_mode: ProcessingMode = ProcessingMode.THOROUGH
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Quality settings
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_gap_tolerance: float = Field(default=0.3, ge=0.0, le=1.0)
    require_validation: bool = True
    
    # Performance settings
    max_concurrent_extractions: int = 5
    max_processing_time_seconds: int = 300
    chunk_size: int = 4000  # tokens
    overlap_size: int = 200  # tokens
    
    # LLM settings
    preferred_provider: str = "anthropic"
    fallback_providers: List[str] = Field(default_factory=lambda: ["openai"])
    model_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Repository analysis
    supported_languages: List[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "java", "go", "rust"]
    )
    max_repository_size_mb: int = 500
    
    # Integration settings
    use_copilot_reasoning: bool = True
    enable_mcp_tools: bool = True
    copilot_confidence_boost: float = 0.1  # Boost confidence when using cognitive approach


class StageProgress(BaseModel):
    """Progress tracking for extraction stages"""
    stage: ExtractionStage
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    def start(self):
        """Mark stage as started"""
        self.status = "running"
        self.started_at = datetime.now()
    
    def complete(self):
        """Mark stage as completed"""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.progress_percentage = 100.0
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000
    
    def fail(self, error_message: str):
        """Mark stage as failed"""
        self.status = "failed"
        self.message = error_message
        self.completed_at = datetime.now()
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000


class ExtractionSession(BaseModel):
    """Session tracking multiple related extractions"""
    session_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    requests: List[str] = Field(default_factory=list)  # request IDs
    results: List[str] = Field(default_factory=list)   # result references
    
    # Session statistics
    total_files_processed: int = 0
    total_extraction_time_ms: float = 0.0
    average_confidence: float = 0.0
    
    # Current state
    active: bool = True
    last_activity: datetime = Field(default_factory=datetime.now)
    
    def add_request(self, request_id: str):
        """Add a request to this session"""
        self.requests.append(request_id)
        self.last_activity = datetime.now()
    
    def update_stats(self, result: ExtractionResult):
        """Update session statistics with new result"""
        self.total_files_processed += len(result.sources_processed)
        self.total_extraction_time_ms += result.processing_time_ms
        
        # Update average confidence
        if self.results:
            current_avg = self.average_confidence
            new_confidence = result.confidence.overall
            count = len(self.results)
            self.average_confidence = (current_avg * count + new_confidence) / (count + 1)
        else:
            self.average_confidence = result.confidence.overall
        
        self.last_activity = datetime.now()


# Protocol definitions for extensibility
class ExtractionStrategy(Protocol):
    """Protocol for extraction strategies"""
    
    async def extract(
        self,
        request: ExtractionRequest[T],
        config: ExtractionConfig
    ) -> ExtractionResult[T]:
        """Execute the extraction strategy"""
        ...
    
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Check if strategy supports the given mode"""
        ...


class ProgressCallback(Protocol):
    """Protocol for progress callbacks"""
    
    async def on_stage_started(self, stage: ExtractionStage, progress: StageProgress):
        """Called when a stage starts"""
        ...
    
    async def on_stage_progress(self, stage: ExtractionStage, progress: StageProgress):
        """Called when stage progress updates"""
        ...
    
    async def on_stage_completed(self, stage: ExtractionStage, progress: StageProgress):
        """Called when a stage completes"""
        ...


# Helper types for repository analysis
@dataclass
class FileAnalysis:
    """Analysis of a single file"""
    path: Path
    size_bytes: int
    language: Optional[str] = None
    encoding: str = "utf-8"
    complexity_score: float = 0.0
    extraction_priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryStructure:
    """Structure analysis of a repository"""
    root_path: Path
    total_files: int
    total_size_bytes: int
    languages: Dict[str, int]  # language -> file count
    file_analyses: List[FileAnalysis] = field(default_factory=list)
    
    # Analysis results
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    recommended_extraction_order: List[Path] = field(default_factory=list)
    estimated_processing_time: float = 0.0