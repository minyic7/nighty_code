"""
LLM-based parser module for handling unsupported or non-conventional file types.
Falls back to LLM when tree-sitter parsers are not available.
"""

from .llm_parser import LLMParser, ParserConfig
from .dynamic_models import (
    # Dynamic models for flexibility
    DynamicEntity,
    DynamicRelationship,
    DynamicFileEntity,
    DynamicExtractionResult,
    DynamicLocation,
    DynamicIdentityCard,
    EntityTypeNormalizer,
    EntityCategory,
    FileType,
    SchemaAdapter
)
# Keep old models for compatibility
from .models import (
    ExtractionRequest,
    ExtractionResponse,
    RepositoryGraphModel,
    CrossFileRelationshipModel
)

__all__ = [
    "LLMParser",
    "ParserConfig",
    # Dynamic models
    "DynamicEntity",
    "DynamicRelationship",
    "DynamicFileEntity",
    "DynamicExtractionResult",
    "DynamicLocation",
    "DynamicIdentityCard",
    "EntityTypeNormalizer",
    "EntityCategory",
    "FileType",
    "SchemaAdapter",
    # Compatibility models
    "ExtractionRequest",
    "ExtractionResponse",
    "RepositoryGraphModel",
    "CrossFileRelationshipModel"
]
