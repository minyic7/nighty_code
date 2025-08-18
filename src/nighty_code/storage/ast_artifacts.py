"""
AST artifact storage for parsed syntax trees.

This module handles saving and managing AST artifacts in various formats
(full, compact, structure-only) for analysis and inspection.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ASTArtifactStorage:
    """
    Manages storage of AST artifacts in different formats.
    """
    
    def __init__(self, base_path: Path = None):
        """
        Initialize AST artifact storage.
        
        Args:
            base_path: Base directory for artifacts (defaults to artifacts/ast_samples)
        """
        if base_path is None:
            base_path = Path('artifacts/ast_samples')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_ast_artifacts(self, ast_json: Dict[str, Any], source_name: str) -> Dict[str, Path]:
        """
        Save AST in multiple formats as artifacts.
        
        Args:
            ast_json: Parsed AST in JSON format
            source_name: Name for the artifact files (usually source file name)
            
        Returns:
            Dictionary mapping format names to saved file paths
        """
        artifacts = {}
        
        # Save full AST
        full_path = self.base_path / f"{source_name}_ast_full.json"
        self._save_json(ast_json, full_path)
        artifacts['full'] = full_path
        
        # Save compact version (without full text)
        compact_ast = self._create_compact_ast(ast_json)
        compact_path = self.base_path / f"{source_name}_ast_compact.json"
        self._save_json(compact_ast, compact_path)
        artifacts['compact'] = compact_path
        
        # Save structure-only version
        structure_ast = self._create_structure_only_ast(ast_json)
        structure_path = self.base_path / f"{source_name}_ast_structure.json"
        self._save_json(structure_ast, structure_path)
        artifacts['structure'] = structure_path
        
        # Save metadata
        metadata = self._create_metadata(ast_json, artifacts)
        metadata_path = self.base_path / f"{source_name}_ast_metadata.json"
        self._save_json(metadata, metadata_path)
        artifacts['metadata'] = metadata_path
        
        return artifacts
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save JSON data to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _create_compact_ast(self, ast_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create compact AST without full text content.
        
        Keeps only essential text for identifiers and literals.
        """
        def compact_node(node: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(node, dict):
                return node
            
            compact = {
                'type': node.get('type'),
                'start_position': node.get('start_position'),
                'end_position': node.get('end_position')
            }
            
            # Keep semantic annotations
            if 'semantic_type' in node:
                compact['semantic_type'] = node['semantic_type']
            if 'is_entity' in node:
                compact['is_entity'] = node['is_entity']
            
            # Only keep text for small, important nodes
            if node.get('type') in ['identifier', 'string_literal', 'integer_literal', 'boolean_literal']:
                if 'text' in node and len(str(node['text'])) <= 100:
                    compact['text'] = node['text']
            
            # Process fields
            if 'fields' in node:
                compact['fields'] = {
                    name: compact_node(field) 
                    for name, field in node['fields'].items()
                }
            
            # Process children
            if 'children' in node:
                compact['children'] = [compact_node(child) for child in node['children']]
            
            return compact
        
        return {
            'file_path': ast_json.get('file_path'),
            'language': ast_json.get('language'),
            'format': 'compact',
            'ast': compact_node(ast_json.get('ast', {}))
        }
    
    def _create_structure_only_ast(self, ast_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create structure-only view showing just types and hierarchy.
        
        Useful for understanding AST shape without details.
        """
        def structure_node(node: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(node, dict):
                return {'type': 'unknown'}
            
            structure = {'type': node.get('type')}
            
            # Mark entities
            if node.get('is_entity'):
                structure['entity'] = node.get('semantic_type', node.get('type'))
                # Add name if it's an identifier in fields
                if 'fields' in node and 'name' in node['fields']:
                    name_field = node['fields']['name']
                    if isinstance(name_field, dict) and 'text' in name_field:
                        structure['name'] = name_field['text']
            
            # Show field names only
            if 'fields' in node:
                structure['fields'] = list(node['fields'].keys())
            
            # Process children
            if 'children' in node:
                structure['children'] = [structure_node(child) for child in node['children']]
            
            return structure
        
        return {
            'file_path': ast_json.get('file_path'),
            'language': ast_json.get('language'),
            'format': 'structure_only',
            'ast': structure_node(ast_json.get('ast', {}))
        }
    
    def _create_metadata(self, ast_json: Dict[str, Any], artifacts: Dict[str, Path]) -> Dict[str, Any]:
        """
        Create metadata about the AST and artifacts.
        """
        def count_nodes(node: Dict[str, Any]) -> Dict[str, int]:
            counts = {}
            if isinstance(node, dict):
                node_type = node.get('type', 'unknown')
                counts[node_type] = counts.get(node_type, 0) + 1
                
                # Count children
                for child in node.get('children', []):
                    child_counts = count_nodes(child)
                    for t, c in child_counts.items():
                        counts[t] = counts.get(t, 0) + c
            
            return counts
        
        # Count entities
        def count_entities(node: Dict[str, Any]) -> list:
            entities = []
            if isinstance(node, dict):
                if node.get('is_entity'):
                    entities.append({
                        'type': node.get('type'),
                        'semantic_type': node.get('semantic_type'),
                        'line': node.get('start_position', {}).get('row')
                    })
                
                for child in node.get('children', []):
                    entities.extend(count_entities(child))
            
            return entities
        
        ast = ast_json.get('ast', {})
        node_counts = count_nodes(ast)
        entities = count_entities(ast)
        
        return {
            'file_path': ast_json.get('file_path'),
            'language': ast_json.get('language'),
            'created_at': datetime.now().isoformat(),
            'statistics': {
                'total_nodes': sum(node_counts.values()),
                'unique_node_types': len(node_counts),
                'total_entities': len(entities),
                'entity_types': list(set(e['semantic_type'] for e in entities if e.get('semantic_type')))
            },
            'node_type_distribution': dict(sorted(node_counts.items())),
            'artifacts': {
                name: str(path) for name, path in artifacts.items()
            }
        }
    
    def load_ast_artifact(self, path: Path) -> Dict[str, Any]:
        """
        Load an AST artifact from file.
        
        Args:
            path: Path to the AST JSON file
            
        Returns:
            Loaded AST dictionary
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_artifacts(self, pattern: str = "*.json") -> list:
        """
        List available AST artifacts.
        
        Args:
            pattern: Glob pattern for filtering files
            
        Returns:
            List of artifact file paths
        """
        return sorted(self.base_path.glob(pattern))