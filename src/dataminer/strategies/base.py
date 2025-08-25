# src/dataminer/strategies/base.py
"""Base extraction strategy and common utilities"""

from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import time

from ..core.types import (
    ExtractionRequest, ExtractionResult, ExtractionConfig,
    ProcessingMode, ExtractionStage, ConfidenceMetrics, GapAnalysis,
    ProgressCallback, StageProgress
)
from ..core.exceptions import ExtractionError, ValidationError, TimeoutError
from ..models.base import ExtractionSchema

# Import LLM types
try:
    from src.llm import LLMManager, Message, MessageRole
    from src.llm.core.types import LLMProvider
except ImportError:
    # Fallback for testing
    LLMManager = None
    Message = None
    MessageRole = None
    LLMProvider = None

# Import MCP types
try:
    from src.mcp import FilesystemServer, ToolCall as MCPToolCall
    from src.mcp.core.types import ToolResult
except ImportError:
    # Fallback for testing
    FilesystemServer = None
    MCPToolCall = None
    ToolResult = None

# Import Copilot types
try:
    from src.copilot import CopilotWorkflow
except ImportError:
    # Fallback for testing
    CopilotWorkflow = None

T = TypeVar('T', bound=ExtractionSchema)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionContext:
    """Context passed between extraction stages"""
    
    # Core data
    request: ExtractionRequest[T]
    config: ExtractionConfig
    
    # Intermediate results
    discovered_content: List[str] = field(default_factory=list)
    raw_extractions: List[Dict[str, Any]] = field(default_factory=list)
    refined_data: Optional[Dict[str, Any]] = None
    
    # Progress tracking
    stage_progress: Dict[ExtractionStage, StageProgress] = field(default_factory=dict)
    current_stage: Optional[ExtractionStage] = None
    
    # Quality metrics
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Integration clients
    llm_client: Optional[Any] = None
    mcp_server: Optional[Any] = None
    copilot_workflow: Optional[Any] = None
    
    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    sources_processed: List[str] = field(default_factory=list)
    
    def add_error(self, error: str, stage: Optional[ExtractionStage] = None):
        """Add an error to the context"""
        if stage:
            error_msg = f"[{stage.value}] {error}"
        else:
            error_msg = error
        self.errors.append(error_msg)
        logger.error(error_msg)
    
    def add_warning(self, warning: str, stage: Optional[ExtractionStage] = None):
        """Add a warning to the context"""
        if stage:
            warning_msg = f"[{stage.value}] {warning}"
        else:
            warning_msg = warning
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)
    
    def get_stage_progress(self, stage: ExtractionStage) -> StageProgress:
        """Get or create progress tracker for a stage"""
        if stage not in self.stage_progress:
            self.stage_progress[stage] = StageProgress(stage=stage)
        return self.stage_progress[stage]
    
    def set_stage_confidence(self, stage: ExtractionStage, confidence: float):
        """Set confidence score for a stage"""
        self.confidence_scores[f"stage_{stage.value}"] = confidence
    
    def get_overall_confidence(self) -> float:
        """Calculate overall confidence from stage scores"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)


class BaseExtractionStrategy(ABC, Generic[T]):
    """Base class for all extraction strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._progress_callbacks: List[ProgressCallback] = []
    
    @abstractmethod
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Check if strategy supports the given processing mode"""
        pass
    
    @abstractmethod
    async def extract(
        self,
        request: ExtractionRequest[T],
        config: ExtractionConfig
    ) -> ExtractionResult[T]:
        """Execute the extraction strategy"""
        pass
    
    def add_progress_callback(self, callback: ProgressCallback):
        """Add a progress callback"""
        self._progress_callbacks.append(callback)
    
    async def _notify_stage_started(self, stage: ExtractionStage, progress: StageProgress):
        """Notify callbacks that a stage started"""
        for callback in self._progress_callbacks:
            try:
                await callback.on_stage_started(stage, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    async def _notify_stage_progress(self, stage: ExtractionStage, progress: StageProgress):
        """Notify callbacks of stage progress"""
        for callback in self._progress_callbacks:
            try:
                await callback.on_stage_progress(stage, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    async def _notify_stage_completed(self, stage: ExtractionStage, progress: StageProgress):
        """Notify callbacks that a stage completed"""
        for callback in self._progress_callbacks:
            try:
                await callback.on_stage_completed(stage, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    async def _initialize_integrations(self, context: ExtractionContext) -> ExtractionContext:
        """Initialize LLM, MCP, and Copilot integrations"""
        try:
            # Initialize LLM client
            # First check if we have an llm_manager attribute set on the strategy
            if hasattr(self, 'llm_manager') and self.llm_manager:
                manager = self.llm_manager
                # Use provider string directly
                context.llm_client = manager.get_client(context.config.preferred_provider)
            elif LLMManager:
                # Fallback: Create LLM manager instance directly
                manager = LLMManager()
                # Use provider string directly
                context.llm_client = manager.get_client(context.config.preferred_provider)
            
            # Initialize MCP filesystem server  
            if FilesystemServer and context.config.enable_mcp_tools:
                context.mcp_server = FilesystemServer()
                await context.mcp_server.initialize()
            
            # Initialize Copilot workflow
            if CopilotWorkflow and context.config.use_copilot_reasoning:
                context.copilot_workflow = CopilotWorkflow()
                await context.copilot_workflow.initialize()
                
        except Exception as e:
            context.add_warning(f"Integration initialization failed: {e}")
        
        return context
    
    async def _validate_request(self, request: ExtractionRequest[T]) -> List[str]:
        """Validate extraction request"""
        issues = []
        
        # Check required fields
        if not request.content:
            issues.append("No content provided for extraction")
        
        if not request.schema_model:
            issues.append("No schema model specified")
        
        # Check configuration
        if request.confidence_threshold < 0 or request.confidence_threshold > 1:
            issues.append("Invalid confidence threshold")
        
        if request.max_iterations < 1:
            issues.append("Max iterations must be at least 1")
        
        # Check content size limits
        if isinstance(request.content, str):
            if len(request.content) > 10_000_000:  # 10MB limit
                issues.append("Content too large for processing")
        elif isinstance(request.content, list):
            total_size = sum(len(str(item)) for item in request.content)
            if total_size > 100_000_000:  # 100MB limit for multiple items
                issues.append("Total content size too large")
        
        return issues
    
    async def _calculate_confidence_metrics(self, context: ExtractionContext, result_data: Optional[T]) -> ConfidenceMetrics:
        """Calculate detailed confidence metrics"""
        metrics = ConfidenceMetrics()
        
        try:
            # Stage-specific confidence scores
            for stage, score in context.confidence_scores.items():
                if stage.startswith("stage_"):
                    stage_enum = ExtractionStage(stage.replace("stage_", ""))
                    metrics.stage_scores[stage_enum] = score
            
            # Calculate component scores
            if result_data:
                # Schema compliance
                validation_result = result_data.validate_completeness()
                metrics.schema_compliance = validation_result.get("completeness_score", 0.0)
                
                # Completeness based on required fields
                required_fields = result_data.get_required_fields()
                complete_fields = sum(1 for field in required_fields 
                                    if getattr(result_data, field, None) is not None)
                metrics.completeness = complete_fields / len(required_fields) if required_fields else 1.0
                
                # Quality based on confidence fields
                confidence_fields = result_data.get_confidence_fields()
                quality_scores = []
                for field in confidence_fields:
                    value = getattr(result_data, field, None)
                    if value is not None:
                        if isinstance(value, (list, dict)):
                            quality_scores.append(1.0 if value else 0.0)
                        elif isinstance(value, str):
                            quality_scores.append(1.0 if value.strip() else 0.0)
                        else:
                            quality_scores.append(1.0)
                    else:
                        quality_scores.append(0.0)
                
                metrics.extraction_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                # Field-level confidence
                for field in required_fields + confidence_fields:
                    value = getattr(result_data, field, None)
                    if value is not None:
                        if isinstance(value, str) and value.strip():
                            metrics.field_confidence[field] = 0.9
                        elif isinstance(value, (list, dict)) and value:
                            metrics.field_confidence[field] = 0.8
                        elif value:
                            metrics.field_confidence[field] = 0.7
                        else:
                            metrics.field_confidence[field] = 0.3
                    else:
                        metrics.field_confidence[field] = 0.0
            
            # Consistency based on errors and warnings
            error_penalty = len(context.errors) * 0.1
            warning_penalty = len(context.warnings) * 0.05
            metrics.consistency = max(0.0, 1.0 - error_penalty - warning_penalty)
            
            # Update overall confidence
            metrics.update_overall()
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {e}")
            # Fallback to basic confidence
            metrics.overall = context.get_overall_confidence()
        
        return metrics
    
    async def _perform_gap_analysis(self, context: ExtractionContext, result_data: Optional[T]) -> GapAnalysis:
        """Perform gap analysis to identify missing data"""
        gap_analysis = GapAnalysis()
        
        try:
            if result_data:
                validation_result = result_data.validate_completeness()
                
                # Missing required fields
                gap_analysis.missing_fields = validation_result.get("missing_required", [])
                
                # Missing confidence fields (incomplete)
                gap_analysis.incomplete_fields = validation_result.get("missing_confidence", [])
                
                # Low confidence fields
                confidence_fields = result_data.get_confidence_fields()
                for field in confidence_fields:
                    value = getattr(result_data, field, None)
                    if value is not None:
                        # Check if field has low quality content
                        if isinstance(value, str) and len(value.strip()) < 10:
                            gap_analysis.low_confidence_fields.append(field)
                        elif isinstance(value, list) and len(value) < 2:
                            gap_analysis.low_confidence_fields.append(field)
                
                # Completeness metrics
                total_fields = len(result_data.get_required_fields())
                missing_count = len(gap_analysis.missing_fields)
                gap_analysis.completeness_score = (total_fields - missing_count) / total_fields if total_fields else 1.0
                gap_analysis.coverage_percentage = gap_analysis.completeness_score * 100
                
                # Generate recommendations
                if gap_analysis.missing_fields:
                    gap_analysis.recommended_actions.append("Provide more comprehensive input data")
                    gap_analysis.recommended_actions.append("Review source content for missing information")
                
                if gap_analysis.incomplete_fields:
                    gap_analysis.recommended_actions.append("Use more detailed extraction prompts")
                    gap_analysis.recommended_actions.append("Increase processing iterations")
                
                if gap_analysis.low_confidence_fields:
                    gap_analysis.recommended_actions.append("Enable cognitive processing mode")
                    gap_analysis.recommended_actions.append("Provide additional context")
                
                # Additional sources suggestions
                if len(context.sources_processed) < 3:
                    gap_analysis.additional_sources.append("Process additional related files")
                
                if not any("README" in source for source in context.sources_processed):
                    gap_analysis.additional_sources.append("Include project README for context")
            
        except Exception as e:
            self.logger.error(f"Error performing gap analysis: {e}")
        
        return gap_analysis
    
    async def _create_result(self, context: ExtractionContext, result_data: Optional[T]) -> ExtractionResult[T]:
        """Create final extraction result"""
        end_time = datetime.now()
        processing_time = (end_time - context.started_at).total_seconds() * 1000
        
        # Calculate confidence and gaps
        confidence_metrics = await self._calculate_confidence_metrics(context, result_data)
        gap_analysis = await self._perform_gap_analysis(context, result_data)
        
        # Determine completed stages
        completed_stages = [
            stage for stage, progress in context.stage_progress.items()
            if progress.status == "completed"
        ]
        
        # Build result
        result = ExtractionResult[T](
            data=result_data,
            raw_extractions=context.raw_extractions,
            confidence=confidence_metrics,
            gap_analysis=gap_analysis,
            stages_completed=completed_stages,
            iterations_used=len(context.raw_extractions),
            processing_time_ms=processing_time,
            sources_processed=context.sources_processed,
            errors=context.errors,
            warnings=context.warnings,
            request_id=context.request.request_id,
            completed_at=end_time
        )
        
        return result
    
    async def _with_timeout(self, coro, timeout_seconds: float, operation_name: str):
        """Execute coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds",
                timeout_seconds=timeout_seconds,
                operation=operation_name
            )
    
    def _chunk_content(self, content: str, chunk_size: int, overlap_size: int = 0) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at word boundaries
            if end < len(content):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:  # Only adjust if reasonably close to end
                    end = start + last_space
                    chunk = content[start:end]
            
            chunks.append(chunk)
            
            # Calculate next start position with overlap
            start = max(start + 1, end - overlap_size)
            if start >= len(content):
                break
        
        return chunks
    
    def _estimate_processing_time(self, content_size: int, mode: ProcessingMode) -> float:
        """Estimate processing time in seconds"""
        base_time = content_size / 10000  # Base time per 10k characters
        
        mode_multipliers = {
            ProcessingMode.FAST: 1.0,
            ProcessingMode.THOROUGH: 2.5,
            ProcessingMode.COGNITIVE: 4.0,
            ProcessingMode.HYBRID: 2.0
        }
        
        return base_time * mode_multipliers.get(mode, 2.0)