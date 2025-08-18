"""
Entity extraction system for different programming languages.

This module provides a framework for extracting entities from source code
with language-specific implementations.
"""

from .base import (
    BaseEntityType, 
    BaseRelationshipType, 
    SourceLocation, 
    BaseEntity, 
    BaseRelationship, 
    BaseFileEntity, 
    ExtractionResult,
    BaseEntityExtractor
)

from .scala_entities import (
    ScalaEntityType,
    ScalaRelationshipType,
    ScalaEntity,
    ScalaFileEntity,
    ScalaEntityExtractor
)

__all__ = [
    # Base types
    'BaseEntityType',
    'BaseRelationshipType', 
    'SourceLocation',
    'BaseEntity',
    'BaseRelationship',
    'BaseFileEntity',
    'ExtractionResult',
    'BaseEntityExtractor',
    
    # Scala-specific
    'ScalaEntityType',
    'ScalaRelationshipType',
    'ScalaEntity', 
    'ScalaFileEntity',
    'ScalaEntityExtractor'
]