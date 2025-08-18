"""
Base entity models for code parsing.

This module defines the minimal core entity types and relationships
that are truly universal across programming languages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from pathlib import Path
from enum import Enum


class BaseEntityType(Enum):
    """Minimal universal entity types that exist in all languages."""
    
    # Core definitions (truly universal)
    FUNCTION = "function"        # Functions/procedures in all languages
    CLASS = "class"             # Classes in OOP languages, structs in others
    VARIABLE = "variable"       # Variables/bindings
    
    # Import/dependency
    IMPORT = "import"           # Import/include/require statements
    
    # Special
    MAIN_ENTRY = "main_entry"   # Main function, entry point
    UNKNOWN = "unknown"         # Fallback for unrecognized entities


class BaseRelationshipType(Enum):
    """Universal relationship types across languages."""
    
    # File-level dependencies
    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    
    # Usage relationships  
    CALLS = "calls"
    REFERENCES = "references"
    
    # Structural relationships
    CONTAINS = "contains"       # File contains class, class contains method
    DEFINES = "defines"         # Package defines class, class defines method
    
    # Inheritance (for OOP languages)
    EXTENDS = "extends"
    IMPLEMENTS = "implements"


@dataclass
class SourceLocation:
    """Location information for an entity in source code."""
    
    file_path: Path
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_start}:{self.column_start}"


@dataclass
class BaseEntity:
    """
    Minimal universal entity representing any code construct.
    
    Language-specific extractors should extend this with their own types.
    """
    
    # Core identity (required)
    entity_type: BaseEntityType  # Can be overridden by language-specific types
    name: str
    qualified_name: str  # Full namespace path
    
    # Location (required)
    location: SourceLocation
    
    # Content (optional)
    signature: Optional[str] = None      # Method signature, class declaration
    text_preview: Optional[str] = None   # First ~100 chars of content
    
    # Language-specific attributes (extensible)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((str(self.entity_type), self.qualified_name, str(self.location.file_path)))


@dataclass
class BaseRelationship:
    """Universal relationship between entities or files."""
    
    relationship_type: BaseRelationshipType
    source: str      # Entity qualified name or file path
    target: str      # Entity qualified name or file path
    
    # Optional context
    location: Optional[SourceLocation] = None
    context: Optional[str] = None
    
    def __hash__(self) -> int:
        return hash((str(self.relationship_type), self.source, self.target))


@dataclass
class BaseFileEntity:
    """Universal file-level entity with extracted components."""
    
    file_path: Path
    language: str
    
    # Extracted entities (language-specific entities should extend BaseEntity)
    entities: List[BaseEntity] = field(default_factory=list)
    relationships: List[BaseRelationship] = field(default_factory=list)
    
    # File-level imports (simplified)
    imports: List[str] = field(default_factory=list)
    
    # Universal metadata
    has_main_entry: bool = False
    is_test_file: bool = False
    
    # Language-specific metadata (extensible)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_main_entities(self) -> List[BaseEntity]:
        """Get entities that define the file's main purpose."""
        # Override in language-specific implementations
        main_types = {BaseEntityType.CLASS, BaseEntityType.MAIN_ENTRY}
        return [e for e in self.entities if e.entity_type in main_types]
    
    def get_dependencies(self) -> Set[str]:
        """Get all external dependencies."""
        deps = set(self.imports)
        for rel in self.relationships:
            if rel.relationship_type in {BaseRelationshipType.IMPORTS, BaseRelationshipType.DEPENDS_ON}:
                deps.add(rel.target)
        return deps


@dataclass
class ExtractionResult:
    """Universal result of entity extraction from a file."""
    
    file_entity: BaseFileEntity
    extraction_time_ms: float
    error: Optional[str] = None
    
    # Statistics
    total_entities: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=dict)  # Using string for language flexibility
    
    def __post_init__(self):
        if self.error is None:
            self.total_entities = len(self.file_entity.entities)
            for entity in self.file_entity.entities:
                entity_type_str = str(entity.entity_type.value) if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                self.entities_by_type[entity_type_str] = self.entities_by_type.get(entity_type_str, 0) + 1


class BaseEntityExtractor:
    """
    Base class for language-specific entity extractors.
    
    Each language should implement its own extractor that extends this.
    """
    
    def __init__(self):
        self.language = "unknown"
    
    def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """
        Extract entities from a file.
        
        Should be implemented by language-specific extractors.
        """
        raise NotImplementedError("Language-specific extractors must implement extract_from_file")
    
    def _create_location(self, file_path: Path, node) -> SourceLocation:
        """Helper to create SourceLocation from tree-sitter node."""
        return SourceLocation(
            file_path=file_path,
            line_start=node.start_point[0] + 1,  # 1-based line numbers
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            column_end=node.end_point[1]
        )
    
    def _get_node_text(self, node, source_code: str, max_length: int = 100) -> str:
        """Helper to extract text from tree-sitter node."""
        text = source_code[node.start_byte:node.end_byte]
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text