"""
Entity artifact storage for extracted code entities.

This module handles saving extracted entities and minimal AST artifacts
for analysis and inspection without the expensive full JSON conversion.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from tree_sitter import Tree
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from ..parsers.model.base import ExtractionResult, BaseEntity, BaseRelationship


class EntityArtifactStorage:
    """
    Manages storage of entity extraction results and minimal AST artifacts.
    """
    
    def __init__(self, base_path: Path = None):
        """
        Initialize entity artifact storage.
        
        Args:
            base_path: Base directory for artifacts (defaults to artifacts/entities)
        """
        if base_path is None:
            base_path = Path('artifacts/entities')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_extraction_artifacts(
        self, 
        extraction_result: ExtractionResult, 
        source_name: str,
        ast_tree=None,
        source_code: str = None
    ) -> Dict[str, Path]:
        """
        Save extraction results and optional AST as artifacts.
        
        Args:
            extraction_result: Result from entity extraction
            source_name: Name for artifact files (usually source file name)
            ast_tree: Optional tree-sitter AST tree
            source_code: Optional source code for AST node text extraction
            
        Returns:
            Dictionary mapping artifact types to saved file paths
        """
        artifacts = {}
        
        # Save extracted entities (JSON)
        entities_path = self.base_path / f"{source_name}_entities.json"
        entities_data = self._serialize_extraction_result(extraction_result)
        self._save_json(entities_data, entities_path)
        artifacts['entities'] = entities_path
        
        # Save minimal AST structure if provided
        if ast_tree and TREE_SITTER_AVAILABLE:
            ast_structure_path = self.base_path / f"{source_name}_ast_structure.json"
            ast_structure = self._create_minimal_ast_structure(ast_tree, source_code)
            self._save_json(ast_structure, ast_structure_path)
            artifacts['ast_structure'] = ast_structure_path
        
        # Save extraction metadata
        metadata_path = self.base_path / f"{source_name}_metadata.json"
        metadata = self._create_extraction_metadata(extraction_result, artifacts)
        self._save_json(metadata, metadata_path)
        artifacts['metadata'] = metadata_path
        
        # Save human-readable summary
        summary_path = self.base_path / f"{source_name}_summary.md"
        summary = self._create_human_summary(extraction_result)
        self._save_text(summary, summary_path)
        artifacts['summary'] = summary_path
        
        return artifacts
    
    def _serialize_extraction_result(self, result: ExtractionResult) -> Dict[str, Any]:
        """Convert extraction result to JSON-serializable format."""
        file_entity = result.file_entity
        
        return {
            'extraction_info': {
                'extraction_time_ms': result.extraction_time_ms,
                'total_entities': result.total_entities,
                'entities_by_type': result.entities_by_type,
                'error': result.error,
                'extracted_at': datetime.now().isoformat()
            },
            'file_info': {
                'file_path': str(file_entity.file_path),
                'language': file_entity.language,
                'package': getattr(file_entity, 'package_name', file_entity.metadata.get('package')),
                'has_main_entry': file_entity.has_main_entry,
                'is_test_file': file_entity.is_test_file
            },
            'imports': file_entity.imports,
            'frameworks_detected': list(getattr(file_entity, 'frameworks_detected', set())),
            'entities': [self._serialize_entity(entity) for entity in file_entity.entities],
            'relationships': [self._serialize_relationship(rel) for rel in file_entity.relationships],
            'dependencies': list(file_entity.get_dependencies()),
            'main_entities': [
                {
                    'type': entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                    'name': entity.name,
                    'qualified_name': entity.qualified_name,
                    'line': entity.location.line_start
                }
                for entity in file_entity.get_main_entities()
            ]
        }
    
    def _serialize_entity(self, entity: BaseEntity) -> Dict[str, Any]:
        """Convert entity to JSON-serializable format."""
        entity_data = {
            'type': entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
            'name': entity.name,
            'qualified_name': entity.qualified_name,
            'location': {
                'file_path': str(entity.location.file_path),
                'line_start': entity.location.line_start,
                'line_end': entity.location.line_end,
                'column_start': entity.location.column_start,
                'column_end': entity.location.column_end
            }
        }
        
        # Add optional fields if present
        if entity.signature:
            entity_data['signature'] = entity.signature
        if entity.text_preview:
            entity_data['text_preview'] = entity.text_preview
        if hasattr(entity, 'attributes') and entity.attributes:
            entity_data['attributes'] = entity.attributes
        
        # Add language-specific attributes
        if hasattr(entity, 'is_case_class'):
            entity_data['is_case_class'] = entity.is_case_class
        if hasattr(entity, 'parameters') and entity.parameters:
            entity_data['parameters'] = entity.parameters
        if hasattr(entity, 'return_type') and entity.return_type:
            entity_data['return_type'] = entity.return_type
        if hasattr(entity, 'parent_types') and entity.parent_types:
            entity_data['parent_types'] = entity.parent_types
        
        return entity_data
    
    def _serialize_relationship(self, relationship: BaseRelationship) -> Dict[str, Any]:
        """Convert relationship to JSON-serializable format."""
        rel_data = {
            'type': relationship.relationship_type.value if hasattr(relationship.relationship_type, 'value') else str(relationship.relationship_type),
            'source': relationship.source,
            'target': relationship.target
        }
        
        if relationship.location:
            rel_data['location'] = {
                'file_path': str(relationship.location.file_path),
                'line_start': relationship.location.line_start
            }
        
        if relationship.context:
            rel_data['context'] = relationship.context
        
        return rel_data
    
    def _create_minimal_ast_structure(self, ast_tree, source_code: str = None) -> Dict[str, Any]:
        """Create minimal AST structure showing only critical nodes."""
        
        def extract_minimal_node(node, depth=0, max_depth=3):
            if depth > max_depth:
                return {'type': node.type, 'truncated': True}
            
            node_data = {
                'type': node.type,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1
            }
            
            # Add text for small, important nodes
            if node.type in ['identifier', 'package_identifier'] and source_code:
                text = source_code[node.start_byte:node.end_byte]
                if len(text) <= 50:
                    node_data['text'] = text
            
            # Add children for critical nodes only
            critical_types = {
                'compilation_unit', 'package_clause', 'import_declaration',
                'object_definition', 'class_definition', 'trait_definition',
                'function_definition', 'template_body'
            }
            
            if node.type in critical_types and node.children:
                children = []
                for child in node.children:
                    if child.type in critical_types or child.type in ['identifier', 'package_identifier']:
                        children.append(extract_minimal_node(child, depth + 1, max_depth))
                
                if children:
                    node_data['children'] = children
            
            return node_data
        
        return {
            'format': 'minimal_ast_structure',
            'created_at': datetime.now().isoformat(),
            'ast': extract_minimal_node(ast_tree.root_node)
        }
    
    def _create_extraction_metadata(self, result: ExtractionResult, artifacts: Dict[str, Path]) -> Dict[str, Any]:
        """Create metadata about the extraction and artifacts."""
        file_entity = result.file_entity
        
        # Count entity types
        entity_type_counts = {}
        for entity in file_entity.entities:
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        return {
            'extraction_metadata': {
                'extraction_time_ms': result.extraction_time_ms,
                'extracted_at': datetime.now().isoformat(),
                'extractor_version': '1.0.0',
                'language': file_entity.language
            },
            'file_analysis': {
                'file_path': str(file_entity.file_path),
                'total_entities': len(file_entity.entities),
                'total_relationships': len(file_entity.relationships),
                'total_imports': len(file_entity.imports),
                'entity_type_distribution': entity_type_counts,
                'frameworks_detected': list(getattr(file_entity, 'frameworks_detected', set())),
                'has_main_entry': file_entity.has_main_entry,
                'is_test_file': file_entity.is_test_file
            },
            'artifacts': {
                name: str(path) for name, path in artifacts.items()
            }
        }
    
    def _create_human_summary(self, result: ExtractionResult) -> str:
        """Create human-readable summary of extraction results."""
        file_entity = result.file_entity
        lines = []
        
        # Header
        lines.append(f"# Entity Extraction Summary")
        lines.append(f"")
        lines.append(f"**File**: {file_entity.file_path}")
        lines.append(f"**Language**: {file_entity.language}")
        lines.append(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Extraction Time**: {result.extraction_time_ms:.1f}ms")
        lines.append(f"")
        
        # File info
        if hasattr(file_entity, 'package_name') and file_entity.package_name:
            lines.append(f"**Package**: {file_entity.package_name}")
        
        frameworks = getattr(file_entity, 'frameworks_detected', set())
        if frameworks:
            lines.append(f"**Frameworks**: {', '.join(frameworks)}")
        
        lines.append(f"**Has Main Entry**: {file_entity.has_main_entry}")
        lines.append(f"")
        
        # Statistics
        lines.append(f"## Statistics")
        lines.append(f"")
        lines.append(f"- **Entities**: {len(file_entity.entities)}")
        lines.append(f"- **Relationships**: {len(file_entity.relationships)}")
        lines.append(f"- **Imports**: {len(file_entity.imports)}")
        lines.append(f"- **Dependencies**: {len(file_entity.get_dependencies())}")
        lines.append(f"")
        
        # Main entities
        main_entities = file_entity.get_main_entities()
        if main_entities:
            lines.append(f"## Main Entities")
            lines.append(f"")
            for entity in main_entities:
                entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                lines.append(f"- **{entity_type}**: `{entity.name}` (line {entity.location.line_start})")
            lines.append(f"")
        
        # Imports
        if file_entity.imports:
            lines.append(f"## Imports")
            lines.append(f"")
            for imp in file_entity.imports:
                lines.append(f"- `{imp}`")
            lines.append(f"")
        
        # All entities
        lines.append(f"## All Entities")
        lines.append(f"")
        for entity in file_entity.entities:
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
            lines.append(f"- **{entity_type}**: `{entity.qualified_name}` (line {entity.location.line_start})")
            if entity.signature:
                lines.append(f"  - Signature: `{entity.signature[:80]}...`")
        
        return '\n'.join(lines)
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save JSON data to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_text(self, text: str, path: Path) -> None:
        """Save text data to file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def load_extraction_result(self, entities_path: Path) -> Dict[str, Any]:
        """
        Load extraction result from entities JSON file.
        
        Args:
            entities_path: Path to the entities JSON file
            
        Returns:
            Loaded extraction data
        """
        with open(entities_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_artifacts(self, pattern: str = "*.json") -> List[Path]:
        """
        List available entity artifacts.
        
        Args:
            pattern: Glob pattern for filtering files
            
        Returns:
            List of artifact file paths
        """
        return sorted(self.base_path.glob(pattern))