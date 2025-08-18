"""
Scala model for comprehensive entity and relationship extraction.

This module provides a complete Scala analysis model that extracts both:
1. Entities (classes, methods, objects, etc.)
2. Relationships (imports, calls, instantiations, etc.)

Key features:
- Variable type tracking for accurate method resolution
- AST-based relationship detection (not regex)
- Automatic deduplication by source location
- Accurate file-to-file dependency tracking
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
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
    """Scala-specific relationship types including cross-file support."""
    
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
    
    # Cross-file relationships
    IMPORTS_TYPE = "imports_type"      # Import resolves to specific entity
    CALLS_METHOD = "calls_method"      # Method call across files
    INSTANTIATES_CLASS = "instantiates_class"  # Class instantiation across files
    USES_TYPE = "uses_type"            # Uses type from another file


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
class VariableTypeInfo:
    """Tracks type information for a variable."""
    
    variable_name: str
    type_name: str  # The class/type of the variable
    qualified_type: str  # Fully qualified type name
    declaration_location: SourceLocation
    scope: str  # The scope where this variable is defined (class/method name)


@dataclass
class ScalaFileEntity(BaseFileEntity):
    """Enhanced Scala file entity with type tracking."""
    
    entities: List[ScalaEntity] = None
    relationships: List[BaseRelationship] = None
    
    # Scala-specific metadata
    package_name: Optional[str] = None
    frameworks_detected: Set[str] = None
    
    # Variable type tracking for accurate resolution
    variable_types: Dict[str, VariableTypeInfo] = None  # variable_name -> type info
    
    # Track instantiations and method calls for cross-file analysis
    instantiations: List[Tuple[str, str, SourceLocation]] = None  # (class_name, qualified_class, location)
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []
        if self.frameworks_detected is None:
            self.frameworks_detected = set()
        if self.variable_types is None:
            self.variable_types = {}
        if self.instantiations is None:
            self.instantiations = []
        
        # Set package in metadata
        if self.package_name:
            self.metadata['package'] = self.package_name
    
    def get_main_entities(self) -> List[ScalaEntity]:
        """Get main Scala entities (objects, classes, traits)."""
        main_types = {ScalaEntityType.OBJECT, ScalaEntityType.CLASS, ScalaEntityType.TRAIT, ScalaEntityType.MAIN_ENTRY}
        return [e for e in self.entities if e.entity_type in main_types]


@dataclass
class CrossFileRelationship:
    """Represents a relationship between entities across files."""
    
    source_entity: str  # Fully qualified name
    target_entity: str  # Fully qualified name
    relationship_type: ScalaRelationshipType
    
    # Context information
    source_context: Dict[str, str]  # file, class, method context
    target_context: Dict[str, str]  # file, class, method context
    
    # Location and evidence
    location: Optional[SourceLocation] = None
    evidence: Optional[str] = None  # Code snippet showing the relationship
    
    def __hash__(self):
        """Hash for deduplication."""
        return hash((self.source_entity, self.target_entity, self.relationship_type.value, 
                    self.location.line_start if self.location else 0))
    
    def __eq__(self, other):
        """Equality for deduplication."""
        if not isinstance(other, CrossFileRelationship):
            return False
        return (self.source_entity == other.source_entity and 
                self.target_entity == other.target_entity and
                self.relationship_type == other.relationship_type and
                (self.location.line_start if self.location else 0) == 
                (other.location.line_start if other.location else 0))


@dataclass 
class RepositoryRelationshipGraph:
    """Complete relationship graph for a collection of files."""
    
    # Cross-file relationships (using set for automatic deduplication)
    cross_file_relationships: Set[CrossFileRelationship] = field(default_factory=set)
    
    # Dependency maps
    file_dependencies: Dict[Path, Set[Path]] = field(default_factory=dict)
    package_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Entity lookup maps (for efficient cross-file resolution)
    entities_by_name: Dict[str, List[ScalaEntity]] = field(default_factory=dict)
    entities_by_fqn: Dict[str, ScalaEntity] = field(default_factory=dict)
    entities_by_file: Dict[Path, List[ScalaEntity]] = field(default_factory=dict)
    
    # Variable type information across files
    variable_types_by_file: Dict[Path, Dict[str, VariableTypeInfo]] = field(default_factory=dict)


class ScalaModelExtractor(BaseEntityExtractor):
    """
    Enhanced Scala model extractor with accurate relationship detection.
    
    Key improvements:
    1. Tracks variable types during AST traversal
    2. Uses AST for relationship detection instead of regex
    3. Deduplicates relationships by source location
    4. Ensures accurate file-to-file dependency tracking
    """
    
    def __init__(self):
        super().__init__()
        self.language = "scala"
        
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-scala is required for Scala model extraction")
        
        self.ts_language = Language(ts_scala.language())
        self.parser = Parser(self.ts_language)
        
        # Framework detection patterns
        self.framework_patterns = {
            'spark': ['org.apache.spark', 'spark.sql', 'spark.streaming'],
            'akka': ['akka.actor', 'akka.stream', 'akka.http'],
            'play': ['play.api', 'play.core'],
            'cats': ['cats.effect', 'cats.implicits'],
            'scalatest': ['org.scalatest', 'scalatest']
        }
    
    def extract_from_file(self, file_path: Path) -> ExtractionResult:
        """Extract complete Scala model from a file."""
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
            
            # Extract entities and relationships in single pass
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
    
    def analyze_repository_relationships(
        self, 
        file_entities: List[ScalaFileEntity]
    ) -> RepositoryRelationshipGraph:
        """
        Analyze cross-file relationships across multiple files.
        
        This is Phase 2 of the extraction - taking per-file results
        and resolving cross-file relationships.
        """
        repo_graph = RepositoryRelationshipGraph()
        
        # Step 1: Build entity and type lookup maps
        self._build_entity_maps(file_entities, repo_graph)
        
        # Step 2: Resolve import relationships
        self._resolve_import_relationships(file_entities, repo_graph)
        
        # Step 3: Resolve instantiation relationships using type info
        self._resolve_instantiation_relationships(file_entities, repo_graph)
        
        # Step 4: Resolve method call relationships using type info
        self._resolve_method_call_relationships(file_entities, repo_graph)
        
        # Step 5: Build dependency maps
        self._build_dependency_maps(repo_graph)
        
        return repo_graph
    
    def _extract_from_ast(self, root_node, source_code: str, file_entity: ScalaFileEntity):
        """Extract entities by recursively walking the AST."""
        
        # Track context as we traverse
        class Context:
            def __init__(self):
                self.package = None
                self.current_class = None  # Current class/object/trait for scoping
                self.parent_entity = None  # Parent entity for nested definitions
                self.current_scope = "global"  # Track current scope for variable types
        
        ctx = Context()
        self._recursive_extract(root_node, source_code, file_entity, ctx)
    
    def _recursive_extract(self, node, source_code: str, file_entity: ScalaFileEntity, ctx):
        """Recursively extract entities from AST nodes with type tracking."""
        
        # Process current node based on its type
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
                old_scope = ctx.current_scope
                ctx.parent_entity = entity
                ctx.current_scope = entity.qualified_name
                
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                
                ctx.parent_entity = old_parent
                ctx.current_scope = old_scope
                return  # Don't recurse again
        
        elif node.type == 'class_definition':
            entity = self._extract_class(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                file_entity.entities.append(entity)
                
                # Set context for nested entities
                old_parent = ctx.parent_entity
                old_scope = ctx.current_scope
                ctx.parent_entity = entity
                ctx.current_scope = entity.qualified_name
                
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                
                ctx.parent_entity = old_parent
                ctx.current_scope = old_scope
                return  # Don't recurse again
        
        elif node.type == 'trait_definition':
            entity = self._extract_trait(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                file_entity.entities.append(entity)
                
                # Set context for nested entities
                old_parent = ctx.parent_entity
                old_scope = ctx.current_scope
                ctx.parent_entity = entity
                ctx.current_scope = entity.qualified_name
                
                # Recurse into children
                for child in node.children:
                    self._recursive_extract(child, source_code, file_entity, ctx)
                
                ctx.parent_entity = old_parent
                ctx.current_scope = old_scope
                return  # Don't recurse again
        
        elif node.type in ['val_definition', 'var_definition']:
            # Track variable types for accurate method resolution
            self._extract_variable_type(node, source_code, file_entity, ctx)
        
        elif node.type == 'function_definition':
            # Build qualified name based on parent
            entity = self._extract_method(node, source_code, file_entity.file_path, ctx.package)
            if entity:
                # Update qualified name if we have a parent
                if ctx.parent_entity:
                    entity.qualified_name = f"{ctx.parent_entity.qualified_name}.{entity.name}"
                file_entity.entities.append(entity)
                
                # Extract method calls within this method using AST
                self._extract_method_calls_from_ast(node, source_code, file_entity, entity.qualified_name)
        
        elif node.type == 'instance_expression':
            # Extract instantiation using AST
            self._process_instantiation(node, source_code, file_entity, ctx)
        
        elif node.type == 'call_expression':
            # Extract method calls using AST
            if ctx.parent_entity:
                self._process_call_expression(node, source_code, file_entity, ctx)
        
        # Recurse into children for all other nodes
        else:
            for child in node.children:
                self._recursive_extract(child, source_code, file_entity, ctx)
    
    def _extract_variable_type(self, node, source_code: str, file_entity: ScalaFileEntity, ctx):
        """Extract variable type information for type tracking."""
        
        # Get variable name
        var_name = None
        for child in node.children:
            if child.type == 'identifier':
                var_name = self._get_node_text(child, source_code, max_length=50).strip()
                break
        
        if not var_name:
            return
        
        # Look for type from instance expression
        type_name = None
        for child in node.children:
            if child.type == 'instance_expression':
                # Extract class name from new expression
                for subchild in child.children:
                    if subchild.type == 'type_identifier':
                        type_name = self._get_node_text(subchild, source_code, max_length=100).strip()
                        break
        
        if type_name and var_name:
            # Store variable type information
            qualified_type = self._resolve_type_name(type_name, file_entity.package_name)
            
            type_info = VariableTypeInfo(
                variable_name=var_name,
                type_name=type_name,
                qualified_type=qualified_type,
                declaration_location=self._create_location(file_entity.file_path, node),
                scope=ctx.current_scope
            )
            
            file_entity.variable_types[var_name] = type_info
    
    def _process_instantiation(self, node, source_code: str, file_entity: ScalaFileEntity, ctx):
        """Process instantiation using AST."""
        
        # Look for type_identifier which contains the class name
        class_name = None
        for child in node.children:
            if child.type == 'type_identifier':
                class_name = self._get_node_text(child, source_code, max_length=100).strip()
                break
        
        if class_name:
            qualified_class = self._resolve_type_name(class_name, file_entity.package_name)
            
            # Store instantiation for cross-file analysis
            file_entity.instantiations.append((
                class_name,
                qualified_class,
                self._create_location(file_entity.file_path, node)
            ))
            
            # Create instantiation relationship
            rel = BaseRelationship(
                relationship_type=ScalaRelationshipType.INSTANTIATES,
                source=ctx.parent_entity.qualified_name if ctx.parent_entity else str(file_entity.file_path),
                target=qualified_class,
                location=self._create_location(file_entity.file_path, node),
                context=f"Instantiates {class_name}"
            )
            file_entity.relationships.append(rel)
    
    def _process_call_expression(self, node, source_code: str, file_entity: ScalaFileEntity, ctx):
        """Process call expression using AST instead of regex."""
        
        call_text = self._get_node_text(node, source_code, max_length=200)
        
        # Look for field_expression which indicates object.method pattern
        for child in node.children:
            if child.type == 'field_expression':
                # Extract object and method from field expression
                object_node = None
                method_node = None
                
                for subchild in child.children:
                    if subchild.type == 'identifier' and object_node is None:
                        object_node = subchild
                    elif subchild.type == 'identifier' and object_node is not None:
                        method_node = subchild
                
                if object_node and method_node:
                    object_name = self._get_node_text(object_node, source_code, max_length=50).strip()
                    method_name = self._get_node_text(method_node, source_code, max_length=50).strip()
                    
                    # Look up the type of the object
                    if object_name in file_entity.variable_types:
                        type_info = file_entity.variable_types[object_name]
                        
                        # Create relationship with accurate type information
                        rel = BaseRelationship(
                            relationship_type=ScalaRelationshipType.CALLS,
                            source=ctx.parent_entity.qualified_name if ctx.parent_entity else str(file_entity.file_path),
                            target=f"{type_info.qualified_type}.{method_name}",
                            location=self._create_location(file_entity.file_path, node),
                            context=f"Calls {method_name} on {object_name} ({type_info.type_name})"
                        )
                        file_entity.relationships.append(rel)
    
    def _resolve_type_name(self, type_name: str, package_name: Optional[str]) -> str:
        """Resolve a type name to its fully qualified form."""
        if '.' in type_name:
            # Already qualified
            return type_name
        elif package_name:
            # Add package qualification
            return f"{package_name}.{type_name}"
        else:
            return type_name
    
    def _extract_method_calls_from_ast(self, method_node, source_code: str, file_entity: ScalaFileEntity, caller_name: str):
        """Extract method calls within a method body using AST traversal."""
        
        # Track seen calls to avoid duplicates (by location)
        seen_locations = set()
        
        for child in self._walk_nodes_recursive(method_node):
            if child.type == 'call_expression':
                location_key = (child.start_point[0], child.start_point[1])
                
                if location_key not in seen_locations:
                    seen_locations.add(location_key)
                    
                    # Process this call expression
                    for subchild in child.children:
                        if subchild.type == 'field_expression':
                            # Extract object.method pattern
                            parts = []
                            for node in subchild.children:
                                if node.type == 'identifier':
                                    parts.append(self._get_node_text(node, source_code, max_length=50).strip())
                            
                            if len(parts) >= 2:
                                object_name = parts[0]
                                method_name = parts[1]
                                
                                # Look up variable type if available
                                target = f"{object_name}.{method_name}"
                                if object_name in file_entity.variable_types:
                                    type_info = file_entity.variable_types[object_name]
                                    target = f"{type_info.qualified_type}.{method_name}"
                                
                                # Create a CALLS relationship
                                rel = BaseRelationship(
                                    relationship_type=ScalaRelationshipType.CALLS,
                                    source=caller_name,
                                    target=target,
                                    location=self._create_location(file_entity.file_path, child),
                                    context=f"Method call"
                                )
                                file_entity.relationships.append(rel)
    
    def _build_entity_maps(self, file_entities: List[ScalaFileEntity], repo_graph: RepositoryRelationshipGraph):
        """Build lookup maps for efficient entity resolution."""
        
        for file_entity in file_entities:
            file_path = file_entity.file_path
            
            # Initialize file entry
            if file_path not in repo_graph.entities_by_file:
                repo_graph.entities_by_file[file_path] = []
            
            # Store variable types for this file
            repo_graph.variable_types_by_file[file_path] = file_entity.variable_types
            
            for entity in file_entity.entities:
                # Index by fully qualified name
                repo_graph.entities_by_fqn[entity.qualified_name] = entity
                
                # Index by simple name (for lookup)
                simple_name = entity.name
                if simple_name not in repo_graph.entities_by_name:
                    repo_graph.entities_by_name[simple_name] = []
                repo_graph.entities_by_name[simple_name].append(entity)
                
                # Index by file
                repo_graph.entities_by_file[file_path].append(entity)
    
    def _resolve_import_relationships(self, file_entities: List[ScalaFileEntity], repo_graph: RepositoryRelationshipGraph):
        """Resolve import statements to actual entities."""
        
        for file_entity in file_entities:
            for import_name in file_entity.imports:
                # Skip external imports (focus on internal project imports)
                if not any(import_name.startswith(pkg) for pkg in ['au.com.cba', file_entity.package_name or '']):
                    continue
                
                target_entity = self._find_entity_by_import(import_name, repo_graph)
                
                if target_entity:
                    # Create cross-file import relationship
                    relationship = CrossFileRelationship(
                        source_entity=str(file_entity.file_path),
                        target_entity=target_entity.qualified_name,
                        relationship_type=ScalaRelationshipType.IMPORTS_TYPE,
                        source_context={
                            "file": file_entity.file_path.name,
                            "package": file_entity.package_name or ""
                        },
                        target_context={
                            "file": self._get_file_for_entity(target_entity, repo_graph).name,
                            "class": target_entity.name,
                            "type": target_entity.entity_type.value
                        },
                        evidence=f"import {import_name}"
                    )
                    
                    repo_graph.cross_file_relationships.add(relationship)
    
    def _resolve_instantiation_relationships(self, file_entities: List[ScalaFileEntity], repo_graph: RepositoryRelationshipGraph):
        """Resolve class instantiation relationships using type information."""
        
        for file_entity in file_entities:
            for class_name, qualified_class, location in file_entity.instantiations:
                # Find the target entity
                target_entity = repo_graph.entities_by_fqn.get(qualified_class)
                
                if not target_entity:
                    # Try by simple name
                    candidates = repo_graph.entities_by_name.get(class_name, [])
                    for candidate in candidates:
                        if candidate.entity_type in [ScalaEntityType.CLASS, ScalaEntityType.CASE_CLASS, ScalaEntityType.OBJECT]:
                            target_entity = candidate
                            break
                
                if target_entity:
                    target_file = self._get_file_for_entity(target_entity, repo_graph)
                    if target_file and target_file != file_entity.file_path:
                        # Cross-file instantiation - find source entity
                        source_entity = self._find_containing_entity(location, file_entity)
                        
                        cross_rel = CrossFileRelationship(
                            source_entity=source_entity.qualified_name if source_entity else str(file_entity.file_path),
                            target_entity=target_entity.qualified_name,
                            relationship_type=ScalaRelationshipType.INSTANTIATES_CLASS,
                            source_context={
                                "file": file_entity.file_path.name,
                                "line": location.line_start
                            },
                            target_context={
                                "file": target_file.name,
                                "class": target_entity.name,
                                "type": target_entity.entity_type.value
                            },
                            location=location
                        )
                        
                        repo_graph.cross_file_relationships.add(cross_rel)
    
    def _resolve_method_call_relationships(self, file_entities: List[ScalaFileEntity], repo_graph: RepositoryRelationshipGraph):
        """Resolve method call relationships using type information."""
        
        for file_entity in file_entities:
            for relationship in file_entity.relationships:
                if relationship.relationship_type == ScalaRelationshipType.CALLS:
                    target = relationship.target
                    
                    # Parse target to get class and method
                    if '.' in target:
                        parts = target.rsplit('.', 1)
                        if len(parts) == 2:
                            target_class, method_name = parts
                            
                            # Find the target entity
                            target_entity = repo_graph.entities_by_fqn.get(target_class)
                            
                            if target_entity:
                                target_file = self._get_file_for_entity(target_entity, repo_graph)
                                if target_file and target_file != file_entity.file_path:
                                    # Cross-file method call
                                    cross_rel = CrossFileRelationship(
                                        source_entity=relationship.source,
                                        target_entity=f"{target_entity.qualified_name}.{method_name}",
                                        relationship_type=ScalaRelationshipType.CALLS_METHOD,
                                        source_context={
                                            "file": file_entity.file_path.name,
                                            "line": relationship.location.line_start if relationship.location else 0
                                        },
                                        target_context={
                                            "file": target_file.name,
                                            "class": target_entity.name,
                                            "method": method_name
                                        },
                                        location=relationship.location
                                    )
                                    
                                    repo_graph.cross_file_relationships.add(cross_rel)
    
    def _find_entity_by_import(self, import_name: str, repo_graph: RepositoryRelationshipGraph) -> Optional[ScalaEntity]:
        """Find an entity that matches an import statement."""
        
        # Direct match by fully qualified name
        if import_name in repo_graph.entities_by_fqn:
            return repo_graph.entities_by_fqn[import_name]
        
        # Try to match by class name (last part of import)
        if '.' in import_name:
            class_name = import_name.split('.')[-1]
            entities = repo_graph.entities_by_name.get(class_name, [])
            for entity in entities:
                if entity.qualified_name == import_name or entity.qualified_name.endswith(f".{class_name}"):
                    return entity
        
        return None
    
    def _find_containing_entity(self, location: SourceLocation, file_entity: ScalaFileEntity) -> Optional[ScalaEntity]:
        """Find the entity that contains a given location."""
        for entity in file_entity.entities:
            if (entity.location.line_start <= location.line_start <= entity.location.line_end):
                return entity
        return None
    
    def _get_file_for_entity(self, entity: ScalaEntity, repo_graph: RepositoryRelationshipGraph) -> Optional[Path]:
        """Get the file path for a given entity."""
        for file_path, entities in repo_graph.entities_by_file.items():
            if entity in entities:
                return file_path
        return None
    
    def _build_dependency_maps(self, repo_graph: RepositoryRelationshipGraph):
        """Build file and package dependency maps - ensuring all cross-file relationships create dependencies."""
        
        for relationship in repo_graph.cross_file_relationships:
            source_file = None
            target_file = None
            
            # Determine source file
            if '/' in relationship.source_entity or '\\' in relationship.source_entity:
                # Source is a file path
                try:
                    source_file = Path(relationship.source_entity)
                except:
                    pass
            else:
                # Source is an entity - find its file
                for file_path, entities in repo_graph.entities_by_file.items():
                    for entity in entities:
                        if entity.qualified_name in relationship.source_entity:
                            source_file = file_path
                            break
                    if source_file:
                        break
            
            # Determine target file
            for file_path, entities in repo_graph.entities_by_file.items():
                for entity in entities:
                    if entity.qualified_name in relationship.target_entity:
                        target_file = file_path
                        break
                if target_file:
                    break
            
            # Add file dependency
            if source_file and target_file and source_file != target_file:
                if source_file not in repo_graph.file_dependencies:
                    repo_graph.file_dependencies[source_file] = set()
                repo_graph.file_dependencies[source_file].add(target_file)
                
                # Package dependency
                source_pkg = relationship.source_context.get("package", "")
                target_pkg = relationship.target_context.get("package", "")
                
                if source_pkg and target_pkg and source_pkg != target_pkg:
                    if source_pkg not in repo_graph.package_dependencies:
                        repo_graph.package_dependencies[source_pkg] = set()
                    repo_graph.package_dependencies[source_pkg].add(target_pkg)
    
    # Utility methods for extraction
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