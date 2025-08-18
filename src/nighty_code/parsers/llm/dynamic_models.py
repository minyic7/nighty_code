"""
Dynamic Pydantic models for flexible artifact representation.
These models allow LLM to define any entity types without rigid enums.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ============================================================================
# OPTIONAL ENUMS (for common cases, but not required)
# ============================================================================

class EntityCategory(str, Enum):
    """High-level categories for grouping entities (optional)."""
    CODE = "code"
    DATA = "data"
    CONFIG = "config"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    SCHEMA = "schema"
    OTHER = "other"


class FileType(str, Enum):
    """Common file types (optional, can use string if not in list)."""
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
    DOCKERFILE = "dockerfile"
    TERRAFORM = "terraform"
    KUBERNETES = "kubernetes"
    CONFIG = "config"
    TEXT = "text"
    UNKNOWN = "unknown"


# ============================================================================
# DYNAMIC MODELS WITH FLEXIBILITY
# ============================================================================

class DynamicLocation(BaseModel):
    """Flexible location information."""
    model_config = ConfigDict(extra='allow')  # Allow additional fields
    
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")
    column_start: Optional[int] = Field(None, description="Starting column")
    column_end: Optional[int] = Field(None, description="Ending column")
    offset: Optional[int] = Field(None, description="Character offset in file")


class DynamicEntity(BaseModel):
    """
    Dynamic entity model that accepts any entity type.
    LLM can define whatever entity types make sense for the file.
    """
    model_config = ConfigDict(extra='allow')  # Allow any additional fields
    
    name: str = Field(description="Name of the entity")
    entity_type: str = Field(description="Type of entity (free-form string)")
    
    # Optional categorization for grouping
    category: Optional[str] = Field(
        None, 
        description="High-level category (code/data/config/etc)"
    )
    
    # Common optional fields
    qualified_name: Optional[str] = Field(None, description="Fully qualified name")
    location: Optional[DynamicLocation] = Field(None, description="Location in source")
    description: Optional[str] = Field(None, description="Description or purpose")
    
    # Flexible metadata - can contain anything
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata LLM wants to include"
    )
    
    # Common code-related fields (optional)
    visibility: Optional[str] = Field(None)
    parameters: Optional[List[Dict[str, Any]]] = Field(None)
    return_type: Optional[str] = Field(None)
    extends: Optional[str] = Field(None)
    implements: Optional[List[str]] = Field(None)
    annotations: Optional[List[str]] = Field(None)
    
    # Data-related fields (optional)
    data_type: Optional[str] = Field(None)
    constraints: Optional[List[str]] = Field(None)
    default_value: Optional[Any] = Field(None)


class DynamicRelationship(BaseModel):
    """
    Dynamic relationship model with flexible types.
    """
    model_config = ConfigDict(extra='allow')
    
    source: str = Field(description="Source entity name or identifier")
    target: str = Field(description="Target entity name or identifier")
    relationship_type: str = Field(description="Type of relationship (free-form)")
    
    # Optional fields
    location: Optional[DynamicLocation] = Field(None)
    confidence: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for LLM-extracted relationships"
    )
    context: Optional[str] = Field(None, description="Additional context")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DynamicFileEntity(BaseModel):
    """
    Flexible file-level model that can represent any file type.
    """
    model_config = ConfigDict(extra='allow')
    
    file_path: str = Field(description="Path to the file")
    file_type: Optional[str] = Field(None, description="Type of file (free-form)")
    
    # Core flexible collections
    entities: List[Union[DynamicEntity, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Entities found in the file"
    )
    relationships: List[Union[DynamicRelationship, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Relationships found in the file"
    )
    
    # Common fields (all optional)
    package: Optional[str] = Field(None)
    namespace: Optional[str] = Field(None)
    imports: List[str] = Field(default_factory=list)
    exports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Flexible metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional file-level metadata"
    )
    
    # Metrics (optional)
    metrics: Optional[Dict[str, Any]] = Field(None)


class DynamicExtractionResult(BaseModel):
    """
    Flexible extraction result that works with any parser.
    """
    model_config = ConfigDict(extra='allow')
    
    success: bool = Field(description="Whether extraction succeeded")
    file_entity: Optional[DynamicFileEntity] = Field(None)
    
    # Raw extracted data (if file_entity conversion fails)
    raw_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw extraction data if structured conversion fails"
    )
    
    # Metadata
    parser_used: str = Field(description="Which parser was used")
    extraction_time_ms: Optional[float] = Field(None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Token usage for LLM
    tokens_used: Optional[int] = Field(None)


# ============================================================================
# NORMALIZATION UTILITIES
# ============================================================================

class EntityTypeNormalizer:
    """
    Optional normalizer for common entity type variations.
    Does NOT fail if type is unknown - just returns as-is.
    """
    
    # Common variations mapping
    COMMON_MAPPINGS = {
        "class": ["Class", "CLASS", "classe", "klass"],
        "function": ["Function", "FUNCTION", "func", "fn", "procedure", "proc"],
        "method": ["Method", "METHOD", "member_function"],
        "variable": ["Variable", "VARIABLE", "var", "field", "attribute"],
        "constant": ["Constant", "CONSTANT", "const", "CONST"],
        "interface": ["Interface", "INTERFACE", "protocol"],
        "table": ["Table", "TABLE", "relation", "entity"],
        "view": ["View", "VIEW", "virtual_table"],
        "index": ["Index", "INDEX", "key"],
        "config": ["Config", "CONFIG", "configuration", "settings"],
        "property": ["Property", "PROPERTY", "prop", "setting"],
        "package": ["Package", "PACKAGE", "namespace", "module"],
        "import": ["Import", "IMPORT", "include", "require"],
        "export": ["Export", "EXPORT", "expose", "public"],
    }
    
    @classmethod
    def normalize(cls, entity_type: str) -> str:
        """
        Normalize entity type to common form.
        Returns original if no mapping found.
        """
        # Check if already normalized
        if entity_type in cls.COMMON_MAPPINGS:
            return entity_type
        
        # Check variations
        for normalized, variations in cls.COMMON_MAPPINGS.items():
            if entity_type in variations or entity_type.lower() in [v.lower() for v in variations]:
                return normalized
        
        # Return as-is if unknown (don't fail!)
        return entity_type.lower()
    
    @classmethod
    def categorize(cls, entity_type: str) -> str:
        """
        Categorize entity type into high-level category.
        """
        code_types = ["class", "function", "method", "interface", "trait", "object", 
                     "variable", "constant", "enum", "type_alias"]
        data_types = ["table", "view", "index", "column", "field", "schema", "database"]
        config_types = ["config", "property", "setting", "parameter", "option"]
        infra_types = ["container", "service", "deployment", "resource", "instance"]
        
        normalized = cls.normalize(entity_type)
        
        if normalized in code_types:
            return EntityCategory.CODE
        elif normalized in data_types:
            return EntityCategory.DATA
        elif normalized in config_types:
            return EntityCategory.CONFIG
        elif normalized in infra_types:
            return EntityCategory.INFRASTRUCTURE
        else:
            return EntityCategory.OTHER


# ============================================================================
# FLEXIBLE IDENTITY CARD MODEL
# ============================================================================

class DynamicIdentityCard(BaseModel):
    """
    Flexible identity card that works with any file type.
    """
    model_config = ConfigDict(extra='allow')
    
    # Core identification
    card_id: str
    file_name: str
    file_path: str
    file_type: Optional[str] = None  # Can be None for unknown types
    
    # Flexible relationships
    upstream_files: List[str] = Field(default_factory=list)
    downstream_files: List[str] = Field(default_factory=list)
    
    # Flexible entities (as dictionaries)
    file_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entities in any format"
    )
    
    # Common fields
    imports: List[str] = Field(default_factory=list)
    exports: List[str] = Field(default_factory=list)
    
    # Flexible metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Summaries
    purpose: Optional[str] = None
    key_functionality: List[str] = Field(default_factory=list)
    llm_summary: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = "3.0.0"  # Dynamic version


# ============================================================================
# USER SCHEMA ADAPTER
# ============================================================================

class SchemaAdapter:
    """
    Adapts extraction results to user-defined schemas.
    Handles mismatches gracefully.
    """
    
    @staticmethod
    def adapt_to_schema(
        data: Dict[str, Any],
        target_schema: type[BaseModel]
    ) -> Optional[BaseModel]:
        """
        Try to adapt extracted data to target schema.
        Returns None if adaptation fails.
        """
        try:
            # Get schema fields
            schema_fields = target_schema.model_fields
            
            # Build adapted data
            adapted = {}
            for field_name, field_info in schema_fields.items():
                # Try different variations of field name
                variations = [
                    field_name,
                    field_name.lower(),
                    field_name.upper(),
                    field_name.replace('_', ''),
                    field_name.replace('_', '-'),
                ]
                
                for variant in variations:
                    if variant in data:
                        adapted[field_name] = data[variant]
                        break
                else:
                    # Use default if available
                    if field_info.default is not None:
                        adapted[field_name] = field_info.default
                    elif field_info.default_factory:
                        adapted[field_name] = field_info.default_factory()
            
            # Try to create instance
            return target_schema(**adapted)
            
        except Exception:
            return None