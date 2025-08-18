"""
Identity card schemas and data models.

This module defines the structure of identity cards that provide
quick file summaries for LLM consumption.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..core.models import FileType, Framework, Complexity, FileClassification


class IdentityCardType(Enum):
    """Types of identity cards."""
    FILE_SUMMARY = "file_summary"
    CODE_ANALYSIS = "code_analysis"
    CONFIG_ANALYSIS = "config_analysis"
    DOCUMENTATION = "documentation"
    DATA_SCHEMA = "data_schema"


@dataclass
class EntityInfo:
    """Simplified entity information for identity cards."""
    
    name: str
    type: str  # class, function, method, trait, object, etc.
    qualified_name: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.type
        }
        if self.qualified_name:
            result["qualified_name"] = self.qualified_name
        if self.line_number:
            result["line_number"] = self.line_number
        return result


@dataclass
class IdentityCard:
    """
    Simplified identity card for a file.
    
    This provides a clear, concise summary that helps LLMs quickly 
    understand a file's content and relationships without parsing 
    the entire content.
    """
    
    # Core identification
    card_id: str
    file_name: str
    file_path: str  # Relative path within repository
    file_type: str  # Language/file type
    
    # File relationships (clear terminology for LLMs)
    upstream_files: List[str] = field(default_factory=list)    # Files this file depends on
    downstream_files: List[str] = field(default_factory=list)  # Files that depend on this file
    
    # File content summary
    file_entities: List[EntityInfo] = field(default_factory=list)  # Classes, functions, etc.
    imports: List[str] = field(default_factory=list)               # Import statements
    exports: List[str] = field(default_factory=list)               # What this file exports/provides
    
    # Metadata
    complexity: str = "unknown"  # low, medium, high
    line_count: Optional[int] = None
    size_bytes: Optional[int] = None
    
    # LLM-optimized summary
    purpose: Optional[str] = None      # One-line description of file's purpose
    key_functionality: List[str] = field(default_factory=list)  # Main features/capabilities
    llm_summary: Optional[str] = None  # AI-generated 2-3 sentence summary
    
    # Optional metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "2.0.0"  # Updated schema version
    
    @classmethod
    def from_classification(
        cls, 
        classification: FileClassification,
        file_path: Optional[Path] = None,
        repository_path: Optional[Path] = None
    ) -> "IdentityCard":
        """
        Create an identity card from a file classification.
        
        Args:
            classification: The file classification
            file_path: Full path to the file
            repository_path: Repository root path
            
        Returns:
            New identity card instance
        """
        
        # Generate card ID
        card_id = cls._generate_card_id(classification.file_path)
        
        # Determine relative path
        relative_path = ""
        if file_path and repository_path:
            try:
                relative_path = str(file_path.relative_to(repository_path))
            except ValueError:
                relative_path = str(file_path)
        elif file_path:
            relative_path = str(file_path)
        else:
            relative_path = str(classification.file_path)
        
        return cls(
            card_id=card_id,
            file_name=classification.file_path.name,
            file_path=relative_path.replace('\\', '/'),  # Use forward slashes
            file_type=classification.classification_result.file_type.value,
            complexity=classification.complexity_level.value,
            size_bytes=classification.metrics.size_bytes,
            line_count=classification.metrics.line_count
        )
    
    @staticmethod
    def _generate_card_id(file_path: Path) -> str:
        """Generate a unique ID for the identity card."""
        import hashlib
        path_str = str(file_path)
        hash_obj = hashlib.sha256(path_str.encode())
        return f"card_{hash_obj.hexdigest()[:12]}"
    
    def add_entity(self, name: str, entity_type: str, 
                   qualified_name: Optional[str] = None,
                   line_number: Optional[int] = None) -> None:
        """
        Add an entity to the file entities list.
        
        Args:
            name: Entity name
            entity_type: Type of entity (class, function, etc.)
            qualified_name: Fully qualified name
            line_number: Line number where entity is defined
        """
        entity = EntityInfo(
            name=name,
            type=entity_type,
            qualified_name=qualified_name,
            line_number=line_number
        )
        self.file_entities.append(entity)
    
    def add_upstream_file(self, file_path: str) -> None:
        """Add a file that this file depends on."""
        if file_path not in self.upstream_files:
            self.upstream_files.append(file_path)
    
    def add_downstream_file(self, file_path: str) -> None:
        """Add a file that depends on this file."""
        if file_path not in self.downstream_files:
            self.downstream_files.append(file_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert identity card to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            # Core identification
            'card_id': self.card_id,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'file_type': self.file_type,
            
            # File relationships
            'upstream_files': self.upstream_files,
            'downstream_files': self.downstream_files,
            
            # File content
            'file_entities': [e.to_dict() for e in self.file_entities],
            'imports': self.imports,
            'exports': self.exports,
            
            # Metadata
            'complexity': self.complexity,
            'line_count': self.line_count,
            'size_bytes': self.size_bytes,
            
            # Summaries
            'purpose': self.purpose,
            'key_functionality': self.key_functionality,
            'llm_summary': self.llm_summary,
            
            # Metadata
            'created_at': self.created_at.isoformat(),
            'version': self.version
        }
    
    def to_llm_prompt(self) -> str:
        """
        Generate a concise text summary optimized for LLM prompts.
        
        Returns:
            String summary that provides key information about the file
        """
        parts = []
        
        # Basic identification
        parts.append(f"File: {self.file_name}")
        parts.append(f"Path: {self.file_path}")
        parts.append(f"Type: {self.file_type}")
        
        # Size and complexity
        if self.line_count:
            parts.append(f"Lines: {self.line_count}")
        parts.append(f"Complexity: {self.complexity}")
        
        # Purpose
        if self.purpose:
            parts.append(f"Purpose: {self.purpose}")
        
        # Dependencies
        if self.upstream_files:
            parts.append(f"Depends on: {len(self.upstream_files)} files")
        if self.downstream_files:
            parts.append(f"Used by: {len(self.downstream_files)} files")
        
        # Entities
        if self.file_entities:
            entity_summary = []
            entity_types = {}
            for entity in self.file_entities:
                entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
            for etype, count in entity_types.items():
                entity_summary.append(f"{count} {etype}(s)")
            parts.append(f"Contains: {', '.join(entity_summary)}")
        
        # Key functionality
        if self.key_functionality:
            parts.append(f"Key features: {', '.join(self.key_functionality[:3])}")
        
        return " | ".join(parts)


# Legacy schemas for backward compatibility (deprecated)
@dataclass
class FileContext:
    """Legacy context information - deprecated, use IdentityCard directly."""
    
    repository_path: Optional[Path] = None
    relative_path: Optional[Path] = None
    git_branch: Optional[str] = None
    git_commit: Optional[str] = None
    
    # File relationships (legacy naming)
    imports_from: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Directory context
    directory_type: Optional[str] = None
    sibling_files: List[str] = field(default_factory=list)


@dataclass
class CodeSummary:
    """Legacy code summary - deprecated, use file_entities in IdentityCard."""
    
    purpose: Optional[str] = None
    functionality: List[str] = field(default_factory=list)
    
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    
    design_patterns: List[str] = field(default_factory=list)
    architectural_patterns: List[str] = field(default_factory=list)
    
    external_dependencies: List[str] = field(default_factory=list)
    internal_imports: List[str] = field(default_factory=list)
    
    has_tests: bool = False
    has_documentation: bool = False
    code_quality_score: Optional[float] = None


@dataclass
class ConfigSummary:
    """Summary of configuration file content."""
    
    config_type: str
    purpose: Optional[str] = None
    
    sections: List[Dict[str, Any]] = field(default_factory=list)
    key_settings: List[Dict[str, Any]] = field(default_factory=list)
    
    environment: Optional[str] = None
    service_name: Optional[str] = None
    
    references_other_configs: List[str] = field(default_factory=list)
    external_resources: List[str] = field(default_factory=list)


@dataclass
class DocumentationSummary:
    """Summary of documentation content."""
    
    doc_type: str
    purpose: Optional[str] = None
    
    sections: List[Dict[str, str]] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    
    code_references: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)
    
    completeness_score: Optional[float] = None
    last_updated: Optional[datetime] = None


@dataclass
class DataSchemaSummary:
    """Summary of data schema or structure."""
    
    schema_type: str
    purpose: Optional[str] = None
    
    tables_or_entities: List[Dict[str, Any]] = field(default_factory=list)
    fields_or_columns: List[Dict[str, Any]] = field(default_factory=list)
    
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    version: Optional[str] = None
    compatibility: List[str] = field(default_factory=list)