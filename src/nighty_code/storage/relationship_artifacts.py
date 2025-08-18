"""
Relationship artifact storage for cross-file relationship analysis.

This module handles saving cross-file relationship graphs and dependency maps
as artifacts for analysis and inspection.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict

from ..parsers.model.scala_model import RepositoryRelationshipGraph, CrossFileRelationship


class RelationshipArtifactStorage:
    """
    Manages storage of cross-file relationship analysis results.
    """
    
    def __init__(self, base_path: Path = None):
        """
        Initialize relationship artifact storage.
        
        Args:
            base_path: Base directory for relationship artifacts
        """
        if base_path is None:
            base_path = Path('artifacts/relationships')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_relationship_graph(
        self, 
        repo_graph: RepositoryRelationshipGraph,
        project_name: str = "project"
    ) -> Dict[str, Path]:
        """
        Save repository relationship graph as artifacts.
        
        Args:
            repo_graph: The complete relationship graph
            project_name: Name to use for artifact files
            
        Returns:
            Dictionary mapping artifact types to file paths
        """
        
        artifacts = {}
        timestamp = datetime.now().isoformat()
        
        # Save cross-file relationships as JSON
        relationships_data = {
            "metadata": {
                "project_name": project_name,
                "generated_at": timestamp,
                "total_cross_file_relationships": len(repo_graph.cross_file_relationships),
                "total_files": len(repo_graph.entities_by_file),
                "total_entities": sum(len(entities) for entities in repo_graph.entities_by_file.values())
            },
            "cross_file_relationships": [
                self._serialize_cross_file_relationship(rel) 
                for rel in repo_graph.cross_file_relationships
            ]
        }
        
        relationships_path = self.base_path / f"{project_name}_relationships.json"
        with open(relationships_path, 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, default=str)
        artifacts["relationships"] = relationships_path
        
        # Save dependency maps
        dependencies_data = {
            "metadata": {
                "project_name": project_name,
                "generated_at": timestamp
            },
            "file_dependencies": {
                str(source): [str(target) for target in targets]
                for source, targets in repo_graph.file_dependencies.items()
            },
            "package_dependencies": {
                source: list(targets)
                for source, targets in repo_graph.package_dependencies.items()
            }
        }
        
        dependencies_path = self.base_path / f"{project_name}_dependencies.json"
        with open(dependencies_path, 'w', encoding='utf-8') as f:
            json.dump(dependencies_data, f, indent=2, default=str)
        artifacts["dependencies"] = dependencies_path
        
        # Save entity lookup maps (for debugging/analysis)
        entity_maps_data = {
            "metadata": {
                "project_name": project_name,
                "generated_at": timestamp
            },
            "entities_by_name": {
                name: [entity.qualified_name for entity in entities]
                for name, entities in repo_graph.entities_by_name.items()
            },
            "entities_by_file": {
                str(file_path): [entity.qualified_name for entity in entities]
                for file_path, entities in repo_graph.entities_by_file.items()
            }
        }
        
        entity_maps_path = self.base_path / f"{project_name}_entity_maps.json"
        with open(entity_maps_path, 'w', encoding='utf-8') as f:
            json.dump(entity_maps_data, f, indent=2, default=str)
        artifacts["entity_maps"] = entity_maps_path
        
        # Generate human-readable summary
        summary_path = self._generate_relationship_summary(repo_graph, project_name)
        artifacts["summary"] = summary_path
        
        return artifacts
    
    def _serialize_cross_file_relationship(self, relationship: CrossFileRelationship) -> Dict[str, Any]:
        """Serialize a CrossFileRelationship to dictionary."""
        
        return {
            "source_entity": relationship.source_entity,
            "target_entity": relationship.target_entity,
            "relationship_type": relationship.relationship_type.value,
            "source_context": relationship.source_context,
            "target_context": relationship.target_context,
            "location": {
                "file": str(relationship.location.file_path) if relationship.location else None,
                "line": relationship.location.line_start if relationship.location else None,
                "column": relationship.location.column_start if relationship.location else None
            } if relationship.location else None,
            "evidence": relationship.evidence
        }
    
    def _generate_relationship_summary(
        self, 
        repo_graph: RepositoryRelationshipGraph, 
        project_name: str
    ) -> Path:
        """Generate human-readable relationship summary."""
        
        summary_path = self.base_path / f"{project_name}_relationship_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Cross-File Relationship Analysis\n\n")
            f.write(f"**Project**: {project_name}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Statistics
            f.write("## Statistics\n\n")
            f.write(f"- **Total Files**: {len(repo_graph.entities_by_file)}\n")
            f.write(f"- **Total Entities**: {sum(len(entities) for entities in repo_graph.entities_by_file.values())}\n")
            f.write(f"- **Cross-File Relationships**: {len(repo_graph.cross_file_relationships)}\n")
            f.write(f"- **File Dependencies**: {len(repo_graph.file_dependencies)}\n")
            f.write(f"- **Package Dependencies**: {len(repo_graph.package_dependencies)}\n\n")
            
            # Relationship type breakdown
            rel_types = {}
            for rel in repo_graph.cross_file_relationships:
                rel_type = rel.relationship_type.value
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            f.write("## Relationship Types\n\n")
            for rel_type, count in sorted(rel_types.items()):
                f.write(f"- **{rel_type}**: {count}\n")
            f.write("\n")
            
            # File dependencies
            if repo_graph.file_dependencies:
                f.write("## File Dependencies\n\n")
                for source_file, target_files in repo_graph.file_dependencies.items():
                    f.write(f"**{source_file.name}** depends on:\n")
                    for target_file in sorted(target_files):
                        f.write(f"- {target_file.name}\n")
                    f.write("\n")
            
            # Package dependencies
            if repo_graph.package_dependencies:
                f.write("## Package Dependencies\n\n")
                for source_pkg, target_pkgs in repo_graph.package_dependencies.items():
                    f.write(f"**{source_pkg}** depends on:\n")
                    for target_pkg in sorted(target_pkgs):
                        f.write(f"- {target_pkg}\n")
                    f.write("\n")
            
            # Sample relationships
            f.write("## Sample Cross-File Relationships\n\n")
            relationships_list = list(repo_graph.cross_file_relationships)
            for i, rel in enumerate(relationships_list[:10]):
                f.write(f"### {i+1}. {rel.relationship_type.value.title()}\n\n")
                f.write(f"**Source**: `{rel.source_entity}`\n")
                f.write(f"**Target**: `{rel.target_entity}`\n")
                f.write(f"**Context**: {rel.source_context.get('file', '')} → {rel.target_context.get('file', '')}\n")
                if rel.evidence:
                    f.write(f"**Evidence**: `{rel.evidence[:100]}{'...' if len(rel.evidence) > 100 else ''}`\n")
                f.write("\n")
        
        return summary_path
    
    def load_relationship_graph(self, project_name: str) -> Dict[str, Any]:
        """
        Load saved relationship graph artifacts.
        
        Args:
            project_name: Name of the project to load
            
        Returns:
            Dictionary containing loaded relationship data
        """
        
        relationships_path = self.base_path / f"{project_name}_relationships.json"
        dependencies_path = self.base_path / f"{project_name}_dependencies.json"
        
        result = {}
        
        if relationships_path.exists():
            with open(relationships_path, 'r', encoding='utf-8') as f:
                result["relationships"] = json.load(f)
        
        if dependencies_path.exists():
            with open(dependencies_path, 'r', encoding='utf-8') as f:
                result["dependencies"] = json.load(f)
        
        return result