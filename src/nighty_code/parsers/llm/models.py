"""
Pydantic models for structured artifact representation.
These models provide a unified schema for both tree-sitter and LLM parsers.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class EntityType(str, Enum):
    """Types of entities that can be extracted from code."""
    CLASS = "class"
    INTERFACE = "interface"
    TRAIT = "trait"
    OBJECT = "object"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"
    IMPORT = "import"
    EXPORT = "export"
    NAMESPACE = "namespace"
    PACKAGE = "package"
    MODULE = "module"
    MAIN_ENTRY = "main_entry"
    CASE_CLASS = "case_class"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    USES = "uses"
    CALLS = "calls"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    IMPORTS = "imports"
    EXPORTS = "exports"
    INSTANTIATES = "instantiates"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    OVERRIDES = "overrides"
    TYPE_OF = "type_of"
    UNKNOWN = "unknown"


class FileType(str, Enum):
    """Supported file types."""
    SCALA = "scala"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    TEXT = "text"
    CONFIG = "config"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Code complexity levels."""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


# ============================================================================
# BASE MODELS
# ============================================================================

class LocationModel(BaseModel):
    """Location information for entities in source code."""
    
    line_start: int = Field(description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")
    column_start: Optional[int] = Field(None, description="Starting column")
    column_end: Optional[int] = Field(None, description="Ending column")
    
    class Config:
        json_schema_extra = {
            "example": {
                "line_start": 10,
                "line_end": 25,
                "column_start": 4,
                "column_end": 30
            }
        }


class EntityModel(BaseModel):
    """Model representing a code entity (class, function, etc.)."""
    
    name: str = Field(description="Name of the entity")
    entity_type: EntityType = Field(description="Type of entity")
    qualified_name: Optional[str] = Field(None, description="Fully qualified name")
    location: Optional[LocationModel] = Field(None, description="Location in source file")
    visibility: Optional[str] = Field(None, description="Visibility modifier (public, private, etc.)")
    is_abstract: bool = Field(False, description="Whether the entity is abstract")
    is_static: bool = Field(False, description="Whether the entity is static")
    annotations: List[str] = Field(default_factory=list, description="Annotations or decorators")
    documentation: Optional[str] = Field(None, description="Associated documentation or comments")
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Parameters for functions/methods")
    return_type: Optional[str] = Field(None, description="Return type for functions/methods")
    extends: Optional[str] = Field(None, description="Parent class or interface")
    implements: List[str] = Field(default_factory=list, description="Implemented interfaces")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "name": "ProcessData",
                "entity_type": "method",
                "qualified_name": "com.example.DataProcessor.ProcessData",
                "location": {"line_start": 15, "line_end": 30},
                "visibility": "public",
                "return_type": "DataFrame"
            }
        }


class RelationshipModel(BaseModel):
    """Model representing a relationship between entities."""
    
    source_entity: str = Field(description="Source entity name or qualified name")
    target_entity: str = Field(description="Target entity name or qualified name")
    relationship_type: RelationshipType = Field(description="Type of relationship")
    location: Optional[LocationModel] = Field(None, description="Location where relationship occurs")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score for LLM-extracted relationships")
    context: Optional[str] = Field(None, description="Additional context about the relationship")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "source_entity": "DataProcessor",
                "target_entity": "DataValidator",
                "relationship_type": "uses",
                "confidence": 0.95
            }
        }


# ============================================================================
# FILE-LEVEL MODELS
# ============================================================================

class FileMetricsModel(BaseModel):
    """Metrics for a source file."""
    
    line_count: int = Field(0, description="Total number of lines")
    size_bytes: int = Field(0, description="File size in bytes")
    comment_lines: Optional[int] = Field(None, description="Number of comment lines")
    blank_lines: Optional[int] = Field(None, description="Number of blank lines")
    code_lines: Optional[int] = Field(None, description="Number of actual code lines")
    
    class Config:
        json_schema_extra = {
            "example": {
                "line_count": 150,
                "size_bytes": 4096,
                "comment_lines": 20,
                "blank_lines": 15,
                "code_lines": 115
            }
        }


class FileEntityModel(BaseModel):
    """Model representing all entities and relationships in a file."""
    
    file_path: str = Field(description="Path to the source file")
    file_type: FileType = Field(description="Type of file")
    package: Optional[str] = Field(None, description="Package or namespace declaration")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    exports: List[str] = Field(default_factory=list, description="Exported entities")
    entities: List[EntityModel] = Field(default_factory=list, description="All entities in the file")
    relationships: List[RelationshipModel] = Field(default_factory=list, description="Relationships within the file")
    metrics: Optional[FileMetricsModel] = Field(None, description="File metrics")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "file_path": "src/main/scala/DataProcessor.scala",
                "file_type": "scala",
                "package": "com.example",
                "imports": ["org.apache.spark.sql._"],
                "entities": [],
                "relationships": []
            }
        }


class ExtractionResultModel(BaseModel):
    """Result of parsing/extracting information from a file."""
    
    success: bool = Field(description="Whether extraction was successful")
    file_entity: Optional[FileEntityModel] = Field(None, description="Extracted file information")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    parser_used: str = Field(description="Which parser was used (tree-sitter or llm)")
    extraction_time_ms: Optional[float] = Field(None, description="Time taken for extraction in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "file_entity": {},
                "errors": [],
                "warnings": [],
                "parser_used": "tree-sitter",
                "extraction_time_ms": 45.3
            }
        }


# ============================================================================
# IDENTITY CARD MODELS
# ============================================================================

class EntityInfoModel(BaseModel):
    """Simplified entity information for identity cards."""
    
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    qualified_name: Optional[str] = Field(None, description="Fully qualified name")
    line_number: Optional[int] = Field(None, description="Line number where entity is defined")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "DataProcessor",
                "type": "class",
                "qualified_name": "com.example.DataProcessor",
                "line_number": 10
            }
        }


class IdentityCardModel(BaseModel):
    """Identity card providing a summary of a file."""
    
    card_id: str = Field(description="Unique identifier for the card")
    file_name: str = Field(description="Name of the file")
    file_path: str = Field(description="Relative path within repository")
    file_type: str = Field(description="Language/file type")
    
    # File relationships
    upstream_files: List[str] = Field(default_factory=list, description="Files this file depends on")
    downstream_files: List[str] = Field(default_factory=list, description="Files that depend on this file")
    
    # Content summary
    file_entities: List[EntityInfoModel] = Field(default_factory=list, description="Key entities in the file")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    exports: List[str] = Field(default_factory=list, description="What this file exports/provides")
    
    # Metadata
    complexity: str = Field("unknown", description="Complexity level")
    line_count: Optional[int] = Field(None, description="Number of lines")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    
    # Summaries
    purpose: Optional[str] = Field(None, description="One-line description of file's purpose")
    key_functionality: List[str] = Field(default_factory=list, description="Main features/capabilities")
    llm_summary: Optional[str] = Field(None, description="AI-generated 2-3 sentence summary")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="When card was created")
    version: str = Field("2.0.0", description="Schema version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "card_id": "card_abc123",
                "file_name": "DataProcessor.scala",
                "file_path": "src/main/scala/DataProcessor.scala",
                "file_type": "scala",
                "upstream_files": ["DataValidator.scala"],
                "downstream_files": ["Main.scala"],
                "complexity": "medium",
                "line_count": 150
            }
        }


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class ClassificationModel(BaseModel):
    """File classification result."""
    
    file_path: str = Field(description="Path to the file")
    file_type: FileType = Field(description="Detected file type")
    complexity: ComplexityLevel = Field(description="Complexity assessment")
    frameworks: List[str] = Field(default_factory=list, description="Detected frameworks")
    patterns: List[str] = Field(default_factory=list, description="Detected design patterns")
    metrics: FileMetricsModel = Field(description="File metrics")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Classification confidence")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "file_path": "src/main/scala/DataProcessor.scala",
                "file_type": "scala",
                "complexity": "medium",
                "frameworks": ["spark", "akka"],
                "patterns": ["singleton", "factory"],
                "metrics": {"line_count": 150, "size_bytes": 4096},
                "confidence": 0.95
            }
        }


# ============================================================================
# REPOSITORY-LEVEL MODELS
# ============================================================================

class CrossFileRelationshipModel(BaseModel):
    """Relationship between entities in different files."""
    
    source_file: str = Field(description="Source file path")
    target_file: str = Field(description="Target file path")
    source_entity: str = Field(description="Entity in source file")
    target_entity: str = Field(description="Entity in target file")
    relationship_type: str = Field(description="Type of relationship")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_file": "DataProcessor.scala",
                "target_file": "DataValidator.scala",
                "source_entity": "DataProcessor",
                "target_entity": "DataValidator",
                "relationship_type": "uses",
                "confidence": 0.95
            }
        }


class RepositoryGraphModel(BaseModel):
    """Repository-wide dependency and relationship graph."""
    
    total_files: int = Field(description="Total number of files analyzed")
    total_entities: int = Field(description="Total entities found")
    file_dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Map of file to its dependencies"
    )
    cross_file_relationships: List[CrossFileRelationshipModel] = Field(
        default_factory=list,
        description="All cross-file relationships"
    )
    entry_points: List[str] = Field(
        default_factory=list,
        description="Identified entry points (main methods, etc.)"
    )
    orphan_files: List[str] = Field(
        default_factory=list,
        description="Files with no dependencies or dependents"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 10,
                "total_entities": 50,
                "file_dependencies": {
                    "Main.scala": ["DataProcessor.scala", "Config.scala"]
                },
                "entry_points": ["Main.scala"],
                "orphan_files": ["README.md"]
            }
        }


# ============================================================================
# EXTRACTION REQUEST/RESPONSE MODELS
# ============================================================================

class ExtractionRequest(BaseModel):
    """Request for extracting structured information."""
    
    file_path: Optional[str] = Field(None, description="Single file to analyze")
    repository_path: Optional[str] = Field(None, description="Repository to analyze")
    target_schema: Optional[Dict[str, Any]] = Field(
        None, 
        description="Target Pydantic schema for extraction"
    )
    use_llm_fallback: bool = Field(
        True, 
        description="Whether to use LLM parser for unsupported files"
    )
    include_relationships: bool = Field(True, description="Extract relationships")
    include_metrics: bool = Field(True, description="Calculate metrics")
    max_files: Optional[int] = Field(None, description="Maximum files to process")
    
    class Config:
        json_schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "use_llm_fallback": True,
                "include_relationships": True,
                "include_metrics": True,
                "max_files": 100
            }
        }


class ExtractionResponse(BaseModel):
    """Response from extraction process."""
    
    success: bool = Field(description="Overall success status")
    files_processed: int = Field(description="Number of files processed")
    files_failed: int = Field(description="Number of files that failed")
    extracted_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Extracted data matching requested schema"
    )
    file_results: List[Union[ExtractionResultModel, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Individual file results"
    )
    repository_graph: Optional[RepositoryGraphModel] = Field(
        None,
        description="Repository-level analysis"
    )
    errors: List[str] = Field(default_factory=list, description="Any errors")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    total_time_ms: Optional[float] = Field(None, description="Total processing time")
    llm_tokens_used: Optional[int] = Field(None, description="Total LLM tokens consumed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "files_processed": 10,
                "files_failed": 0,
                "extracted_data": {},
                "file_results": [],
                "total_time_ms": 1234.5,
                "llm_tokens_used": 5000
            }
        }