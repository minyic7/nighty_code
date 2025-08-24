# src/dataminer/models/base.py
"""Base models for data extraction schemas"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class ExtractionSchema(BaseModel, ABC):
    """Base class for all extraction schemas"""
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Don't allow extra fields
    
    @abstractmethod
    def get_confidence_fields(self) -> List[str]:
        """Return list of fields that contribute to confidence scoring"""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for completeness check"""
        pass
    
    def validate_completeness(self) -> Dict[str, Any]:
        """Validate schema completeness and return metrics"""
        required_fields = self.get_required_fields()
        confidence_fields = self.get_confidence_fields()
        
        missing_required = []
        missing_confidence = []
        
        for field_name in required_fields:
            value = getattr(self, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_required.append(field_name)
        
        for field_name in confidence_fields:
            value = getattr(self, field_name, None)
            if value is None or (isinstance(value, (list, dict)) and not value):
                missing_confidence.append(field_name)
        
        total_fields = len(required_fields)
        complete_fields = total_fields - len(missing_required)
        completeness_score = complete_fields / total_fields if total_fields > 0 else 1.0
        
        return {
            "completeness_score": completeness_score,
            "missing_required": missing_required,
            "missing_confidence": missing_confidence,
            "total_required_fields": total_fields,
            "complete_required_fields": complete_fields
        }


class BaseExtractedData(ExtractionSchema):
    """Base class for extracted data with common metadata"""
    
    # Extraction metadata
    extracted_at: datetime = Field(default_factory=datetime.now)
    source_files: List[str] = Field(default_factory=list, description="Source files this data was extracted from")
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence in extraction")
    
    # Content metadata
    language: Optional[str] = Field(None, description="Programming language or format")
    version: Optional[str] = Field(None, description="Version information if available")
    encoding: str = Field(default="utf-8", description="Character encoding")
    
    # Additional context
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    notes: List[str] = Field(default_factory=list, description="Additional notes from extraction")
    
    def get_confidence_fields(self) -> List[str]:
        """Common confidence fields for all extracted data"""
        return ["source_files", "language", "tags"]
    
    def get_required_fields(self) -> List[str]:
        """Common required fields"""
        return ["extracted_at", "source_files"]
    
    def add_source_file(self, file_path: str):
        """Add a source file to the list"""
        if file_path not in self.source_files:
            self.source_files.append(file_path)
    
    def add_tag(self, tag: str):
        """Add a tag if not already present"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_note(self, note: str):
        """Add a note"""
        self.notes.append(note)


class NestedExtractionSchema(BaseExtractedData):
    """Base for schemas that support nested/hierarchical extraction"""
    
    # Hierarchical information
    parent_id: Optional[str] = Field(None, description="ID of parent element")
    children_ids: List[str] = Field(default_factory=list, description="IDs of child elements")
    depth_level: int = Field(default=0, ge=0, description="Nesting depth level")
    
    # Relationship metadata
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other elements")
    references: List[str] = Field(default_factory=list, description="References to other elements")
    
    def get_confidence_fields(self) -> List[str]:
        """Extend base confidence fields with hierarchical fields"""
        base_fields = super().get_confidence_fields()
        return base_fields + ["dependencies", "references"]
    
    def add_child(self, child_id: str):
        """Add a child ID"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def add_dependency(self, dependency: str):
        """Add a dependency"""
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)
    
    def add_reference(self, reference: str):
        """Add a reference"""
        if reference not in self.references:
            self.references.append(reference)


class ValidationResult(BaseModel):
    """Result of schema validation"""
    
    is_valid: bool = Field(description="Whether the schema is valid")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Completeness score")
    
    # Detailed validation results
    missing_required_fields: List[str] = Field(default_factory=list)
    missing_optional_fields: List[str] = Field(default_factory=list)
    invalid_fields: List[Dict[str, str]] = Field(default_factory=list)
    
    # Suggestions
    improvement_suggestions: List[str] = Field(default_factory=list)
    alternative_approaches: List[str] = Field(default_factory=list)
    
    # Metadata
    validated_at: datetime = Field(default_factory=datetime.now)
    validator_version: Optional[str] = None


class ExtractionMetrics(BaseModel):
    """Metrics for extraction performance and quality"""
    
    # Timing metrics
    total_time_ms: float = Field(ge=0.0, description="Total extraction time")
    llm_time_ms: float = Field(default=0.0, ge=0.0, description="Time spent in LLM calls")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Time spent in processing")
    
    # Quality metrics
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Overall extraction confidence")
    schema_compliance: float = Field(ge=0.0, le=1.0, description="Schema compliance score")
    completeness: float = Field(ge=0.0, le=1.0, description="Data completeness score")
    
    # Resource usage
    tokens_used: int = Field(default=0, ge=0, description="Total tokens used")
    api_calls_made: int = Field(default=0, ge=0, description="Number of API calls")
    files_processed: int = Field(default=0, ge=0, description="Number of files processed")
    
    # Error tracking
    errors_encountered: int = Field(default=0, ge=0)
    warnings_generated: int = Field(default=0, ge=0)
    retries_attempted: int = Field(default=0, ge=0)
    
    # Additional context
    extraction_mode: Optional[str] = None
    provider_used: Optional[str] = None
    model_used: Optional[str] = None


class ProgressTracker(BaseModel):
    """Track progress through multi-stage extraction"""
    
    current_stage: str = Field(description="Current extraction stage")
    completed_stages: List[str] = Field(default_factory=list)
    total_stages: int = Field(ge=1, description="Total number of stages")
    
    # Progress information
    overall_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    stage_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    # Messages
    current_message: str = Field(default="Starting extraction...")
    recent_messages: List[str] = Field(default_factory=list)
    
    def advance_stage(self, new_stage: str, message: Optional[str] = None):
        """Advance to next stage"""
        if self.current_stage not in self.completed_stages:
            self.completed_stages.append(self.current_stage)
        
        self.current_stage = new_stage
        self.overall_progress = (len(self.completed_stages) / self.total_stages) * 100
        self.stage_progress = 0.0
        
        if message:
            self.update_message(message)
    
    def update_progress(self, stage_progress: float, message: Optional[str] = None):
        """Update current stage progress"""
        self.stage_progress = min(stage_progress, 100.0)
        
        # Update overall progress
        base_progress = (len(self.completed_stages) / self.total_stages) * 100
        stage_contribution = (self.stage_progress / self.total_stages)
        self.overall_progress = min(base_progress + stage_contribution, 100.0)
        
        if message:
            self.update_message(message)
    
    def update_message(self, message: str):
        """Update current message and add to recent messages"""
        self.current_message = message
        self.recent_messages.append(f"{datetime.now().strftime('%H:%M:%S')}: {message}")
        
        # Keep only last 10 messages
        if len(self.recent_messages) > 10:
            self.recent_messages = self.recent_messages[-10:]