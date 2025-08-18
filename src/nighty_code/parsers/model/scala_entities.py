"""
Scala-specific entity types and extractor.

This module extends the base entity system with Scala-specific
constructs and provides extraction from Scala AST.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from pathlib import Path
from enum import Enum

try:
    import tree_sitter_scala as ts_scala
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .base import (
    BaseEntityType, BaseRelationshipType, BaseEntity, BaseRelationship, 
    BaseFileEntity, BaseEntityExtractor, ExtractionResult, SourceLocation
)


class ScalaEntityType(Enum):
    """Scala-specific entity types including base types."""
    
    # Base types (included)
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    MAIN_ENTRY = "main_entry"
    UNKNOWN = "unknown"
    
    # Scala-specific definitions
    OBJECT = "object"           # Scala singleton objects
    TRAIT = "trait"             # Scala traits
    CASE_CLASS = "case_class"   # Scala case classes
    
    # Scala variables
    VAL = "val"                 # Immutable values
    VAR = "var"                 # Mutable variables
    
    # Scala methods
    METHOD = "method"           # Class/object methods
    DEF = "def"                 # Function definitions
    
    # Package and imports
    PACKAGE = "package"         # Package declarations


class ScalaRelationshipType(Enum):
    """Scala-specific relationship types including base types."""
    
    # Base types (included)
    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    CALLS = "calls"
    REFERENCES = "references"
    CONTAINS = "contains"
    DEFINES = "defines"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    
    # Scala-specific relationships
    WITH = "with"              # Trait mixing
    INSTANTIATES = "instantiates"  # Object instantiation
    APPLIES = "applies"            # Function application


@dataclass
class ScalaEntity(BaseEntity):
    """Scala-specific entity with additional attributes."""
    
    # Scala-specific attributes (all with defaults)
    is_case_class: bool = False
    is_abstract: bool = False
    is_sealed: bool = False
    is_implicit: bool = False
    is_lazy: bool = False
    
    # Method/function specific
    parameters: List[str] = None
    return_type: Optional[str] = None
    
    # Class/trait specific  
    parent_types: List[str] = None
    type_parameters: List[str] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.parent_types is None:
            self.parent_types = []
        if self.type_parameters is None:
            self.type_parameters = []


@dataclass
class ScalaFileEntity(BaseFileEntity):
    """Scala-specific file entity."""
    
    entities: List[ScalaEntity] = None
    relationships: List[BaseRelationship] = None
    
    # Scala-specific metadata
    package_name: Optional[str] = None
    frameworks_detected: Set[str] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []
        if self.frameworks_detected is None:
            self.frameworks_detected = set()
        
        # Set package in metadata
        if self.package_name:
            self.metadata['package'] = self.package_name
    
    def get_main_entities(self) -> List[ScalaEntity]:
        """Get main Scala entities (objects, classes, traits)."""
        main_types = {ScalaEntityType.OBJECT, ScalaEntityType.CLASS, ScalaEntityType.TRAIT, ScalaEntityType.MAIN_ENTRY}
        return [e for e in self.entities if e.entity_type in main_types]


class ScalaEntityExtractor(BaseEntityExtractor):
    """
    Extracts entities from Scala source code using tree-sitter.
    
    Focuses on key entities needed for file understanding and cross-file relationships.
    """
    
    def __init__(self):
        super().__init__()
        self.language = "scala"
        
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-scala is required for Scala entity extraction")
        
        self.ts_language = Language(ts_scala.language())
        self.parser = Parser(self.ts_language)
        
        # Define critical node types we care about
        self.critical_node_types = {
            'package_clause',
            'import_declaration', 
            'object_definition',
            'class_definition',
            'trait_definition',
            'function_definition',  # methods and functions
            'val_definition',
            'var_definition',
            'import_statement',  # Internal imports like import spark.implicits._
            'call_expression',   # Method calls
            'field_expression'   # Field accesses like config.getString()
        }
        
        # Framework detection patterns
        self.framework_patterns = {
            'spark': ['org.apache.spark', 'spark.sql', 'spark.streaming'],
            'akka': ['akka.actor', 'akka.stream', 'akka.http'],
            'play': ['play.api', 'play.core'],
            'cats': ['cats.effect', 'cats.implicits'],
            'scalatest': ['org.scalatest', 'scalatest']
        }
    
    def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """Extract entities from a Scala file."""
        start_time = time.time()
        
        try:
            # Read source
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse with tree-sitter
            tree = self.parser.parse(bytes(source_code, 'utf-8'))
            
            # Create file entity
            file_entity = ScalaFileEntity(
                file_path=file_path,
                language=self.language
            )
            
            # Extract entities and relationships
            self._extract_from_ast(tree.root_node, source_code, file_entity)
            
            # Detect frameworks
            self._detect_frameworks(file_entity)
            
            # Calculate extraction time
            extraction_time = (time.time() - start_time) * 1000
            
            return ExtractionResult(
                file_entity=file_entity,
                extraction_time_ms=extraction_time
            )
            
        except Exception as e:
            extraction_time = (time.time() - start_time) * 1000
            return ExtractionResult(
                file_entity=ScalaFileEntity(file_path=file_path, language=self.language),
                extraction_time_ms=extraction_time,
                error=str(e)
            )
    
    def _extract_from_ast(self, root_node, source_code: str, file_entity: ScalaFileEntity):
        """Extract entities by recursively walking the AST."""
        
        # Track context as we traverse
        class Context:
            def __init__(self):
                self.package = None
                self.current_class = None  # Current class/object/trait for scoping
                self.parent_entity = None  # Parent entity for nested definitions
        
        ctx = Context()
        self._recursive_extract(root_node, source_code, file_entity, ctx)
    
    def _recursive_extract(self, node, source_code: str, file_entity: ScalaFileEntity, ctx):
        """Recursively extract entities from AST nodes."""
        
        # Process current node if it's a critical type
        if node.type == 'package_clause':
            package_name = self._extract_package_name(node, source_code)
            if package_name:
                ctx.package = package_name
                file_entity.package_name = package_name
        
        elif node.type in ['import_declaration', 'import_statement']:
            import_name = self._extract_import(node, source_code)
            if import_name and import_name not in file_entity.imports:
                file_entity.imports.append(import_name)
                # Determine source for the import relationship
                source = ctx.parent_entity.qualified_name if ctx.parent_entity else str(file_entity.file_path)
                context = f"Internal import in {ctx.parent_entity.name}" if ctx.parent_entity else None
                
                rel = BaseRelationship(
                    relationship_type=ScalaRelationshipType.IMPORTS,
                    source=source,
                    target=import_name,
                    location=self._create_location(file_entity.file_path, node),
                    context=context
                )
                file_entity.relationships.append(rel)
        
        elif node.type == 'object_definition':
            entity = self._extract_object(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                file_entity.entities.append(entity)
                if self._has_main_method(node, source_code):
                    file_entity.has_main_entry = True
                
                # Set context for nested entities
                old_parent = ctx.parent_entity
                ctx.parent_entity = entity
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                ctx.parent_entity = old_parent
                return  # Don't recurse again
        
        elif node.type == 'class_definition':
            entity = self._extract_class(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                file_entity.entities.append(entity)
                
                # Set context for nested entities
                old_parent = ctx.parent_entity
                ctx.parent_entity = entity
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                ctx.parent_entity = old_parent
                return  # Don't recurse again
        
        elif node.type == 'trait_definition':
            entity = self._extract_trait(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                file_entity.entities.append(entity)
                
                # Set context for nested entities
                old_parent = ctx.parent_entity
                ctx.parent_entity = entity
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                ctx.parent_entity = old_parent
                return  # Don't recurse again
        
        elif node.type == 'function_definition':
            # Build qualified name based on parent
            entity = self._extract_method(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                # Update qualified name if we have a parent
                if ctx.parent_entity:
                    entity.qualified_name = f"{ctx.parent_entity.qualified_name}.{entity.name}"
                file_entity.entities.append(entity)
                
                # Extract method calls
                self._extract_method_calls(node, source_code, file_entity, entity.qualified_name)
        
        elif node.type in ['val_definition', 'var_definition']:
            # Could extract these as variables if needed
            pass
        
        elif node.type == 'call_expression':
            # Track method calls if we're inside a method
            pass
        
        # Recurse into children for all other nodes
        else:
            for child in node.children:
                self._recursive_extract(child, source_code, file_entity, ctx)
    
    def _walk_critical_nodes(self, node):
        """Walk only critical nodes to avoid expensive full traversal."""
        if node.type in self.critical_node_types:
            yield node
        
        # Always recurse into children for structural nodes
        structural_nodes = [
            'compilation_unit', 'template_body', 'block',
            'class_body', 'object_body', 'trait_body'
        ]
        
        if node.type in structural_nodes or node.type not in self.critical_node_types:
            for child in node.children:
                yield from self._walk_critical_nodes(child)
    
    def _extract_package_name(self, node, source_code: str) -> Optional[str]:
        """Extract package name from package_clause node."""
        name_node = node.child_by_field_name('name')
        if name_node:
            return self._get_node_text(node, source_code).replace('package ', '').strip()
        return None
    
    def _extract_import(self, node, source_code: str) -> Optional[str]:
        """Extract import path from import_declaration or import_statement node."""
        import_text = self._get_node_text(node, source_code)
        # Remove 'import' keyword and clean up
        import_path = import_text.replace('import ', '').strip()
        return import_path if import_path else None
    
    def _extract_object(self, node, source_code: str, file_path: Path, package: Optional[str]) -> Optional[ScalaEntity]:
        """Extract Scala object definition."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code, max_length=50)
        qualified_name = f"{package}.{name}" if package else name
        
        # Get signature (first line of object definition)
        signature = self._get_node_text(node, source_code, max_length=200).split('\n')[0]
        
        return ScalaEntity(
            entity_type=ScalaEntityType.OBJECT,
            name=name,
            qualified_name=qualified_name,
            location=self._create_location(file_path, node),
            signature=signature,
            text_preview=self._get_node_text(node, source_code, max_length=100)
        )
    
    def _extract_class(self, node, source_code: str, file_path: Path, package: Optional[str]) -> Optional[ScalaEntity]:
        """Extract Scala class definition."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code, max_length=50)
        qualified_name = f"{package}.{name}" if package else name
        
        # Check if it's a case class
        full_text = self._get_node_text(node, source_code, max_length=200)
        is_case_class = 'case class' in full_text
        
        signature = full_text.split('\n')[0] if '\n' in full_text else full_text
        
        return ScalaEntity(
            entity_type=ScalaEntityType.CASE_CLASS if is_case_class else ScalaEntityType.CLASS,
            name=name,
            qualified_name=qualified_name,
            location=self._create_location(file_path, node),
            signature=signature,
            is_case_class=is_case_class,
            text_preview=self._get_node_text(node, source_code, max_length=100)
        )
    
    def _extract_trait(self, node, source_code: str, file_path: Path, package: Optional[str]) -> Optional[ScalaEntity]:
        """Extract Scala trait definition."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code, max_length=50)
        qualified_name = f"{package}.{name}" if package else name
        
        signature = self._get_node_text(node, source_code, max_length=200).split('\n')[0]
        
        return ScalaEntity(
            entity_type=ScalaEntityType.TRAIT,
            name=name,
            qualified_name=qualified_name,
            location=self._create_location(file_path, node),
            signature=signature,
            text_preview=self._get_node_text(node, source_code, max_length=100)
        )
    
    def _extract_method(self, node, source_code: str, file_path: Path, package: Optional[str]) -> Optional[ScalaEntity]:
        """Extract method/function definition."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, source_code, max_length=50)
        
        # Build qualified name (need to find containing class/object)
        qualified_name = name  # Simplified for now
        
        # Check if it's main method
        entity_type = ScalaEntityType.MAIN_ENTRY if name == 'main' else ScalaEntityType.METHOD
        
        signature = self._get_node_text(node, source_code, max_length=200).split('\n')[0]
        
        return ScalaEntity(
            entity_type=entity_type,
            name=name,
            qualified_name=qualified_name,
            location=self._create_location(file_path, node),
            signature=signature,
            text_preview=self._get_node_text(node, source_code, max_length=100)
        )
    
    def _has_main_method(self, object_node, source_code: str) -> bool:
        """Check if object has a main method."""
        object_text = self._get_node_text(object_node, source_code)
        return 'def main(' in object_text
    
    def _extract_method_calls(self, method_node, source_code: str, file_entity: ScalaFileEntity, caller_name: str):
        """Extract method calls within a method body."""
        seen_calls = set()  # Track to avoid duplicates
        
        for child in self._walk_nodes_recursive(method_node):
            if child.type == 'call_expression':
                # Get the full call expression text
                call_text = self._get_node_text(child, source_code, max_length=200)
                
                # Extract just the method chain before arguments
                if '(' in call_text:
                    method_chain = call_text.split('(')[0].strip()
                    
                    # Skip if we've already seen this exact call at this location
                    call_key = f"{method_chain}:{child.start_point}"
                    if call_key not in seen_calls:
                        seen_calls.add(call_key)
                        
                        # Create a CALLS relationship
                        rel = BaseRelationship(
                            relationship_type=ScalaRelationshipType.CALLS,
                            source=caller_name,
                            target=method_chain,
                            location=self._create_location(file_entity.file_path, child),
                            context=f"Method call"
                        )
                        file_entity.relationships.append(rel)
    
    def _walk_nodes_recursive(self, node):
        """Recursively walk all nodes (for method body analysis)."""
        yield node
        for child in node.children:
            yield from self._walk_nodes_recursive(child)
    
    def _detect_frameworks(self, file_entity: ScalaFileEntity):
        """Detect frameworks used based on imports."""
        for import_name in file_entity.imports:
            for framework, patterns in self.framework_patterns.items():
                if any(pattern in import_name for pattern in patterns):
                    file_entity.frameworks_detected.add(framework)