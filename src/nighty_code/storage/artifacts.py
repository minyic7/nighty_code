"""
Artifact storage and management for identity cards.

This module provides classes for saving, loading, and managing
identity cards and classification results as persistent artifacts.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging

from ..identity.schemas import IdentityCard
from ..core.models import FileClassification

logger = logging.getLogger(__name__)


class ArtifactStorage:
    """
    Handles storage and retrieval of identity card artifacts.
    
    Supports multiple output formats and provides versioning
    and metadata management for stored artifacts.
    """
    
    def __init__(self, base_path: Path, create_dirs: bool = True):
        """
        Initialize artifact storage.
        
        Args:
            base_path: Base directory for storing artifacts
            create_dirs: Whether to create directories if they don't exist
        """
        self.base_path = Path(base_path)
        
        if create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        self.cards_dir = self.base_path / "identity_cards"
        self.classifications_dir = self.base_path / "classifications"
        self.reports_dir = self.base_path / "reports"
        
        if create_dirs:
            self.cards_dir.mkdir(exist_ok=True)
            self.classifications_dir.mkdir(exist_ok=True)
            self.reports_dir.mkdir(exist_ok=True)
    
    def save_identity_cards(
        self, 
        cards: List[IdentityCard], 
        collection_name: str,
        format: str = "json"
    ) -> Path:
        """
        Save a collection of identity cards.
        
        Args:
            cards: List of identity cards to save
            collection_name: Name for this collection of cards
            format: Output format ('json', 'yaml', 'markdown')
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"{collection_name}_{timestamp}.json"
            filepath = self.cards_dir / filename
            self._save_cards_json(cards, filepath)
            
        elif format.lower() == "yaml":
            filename = f"{collection_name}_{timestamp}.yaml"
            filepath = self.cards_dir / filename
            self._save_cards_yaml(cards, filepath)
            
        elif format.lower() == "markdown":
            filename = f"{collection_name}_{timestamp}.md"
            filepath = self.cards_dir / filename
            self._save_cards_markdown(cards, filepath)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(cards)} identity cards to {filepath}")
        return filepath
    
    def save_classifications(
        self,
        classifications: List[FileClassification],
        collection_name: str,
        format: str = "json"
    ) -> Path:
        """
        Save classification results.
        
        Args:
            classifications: List of file classifications
            collection_name: Name for this collection
            format: Output format ('json', 'yaml')
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{collection_name}_classifications_{timestamp}.{format}"
        filepath = self.classifications_dir / filename
        
        # Convert to serializable format
        data = {
            'metadata': {
                'collection_name': collection_name,
                'created_at': datetime.now().isoformat(),
                'total_files': len(classifications),
                'version': '1.0.0'
            },
            'classifications': [self._classification_to_dict(c) for c in classifications]
        }
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(classifications)} classifications to {filepath}")
        return filepath
    
    def load_identity_cards(self, filepath: Path) -> List[IdentityCard]:
        """
        Load identity cards from a file.
        
        Args:
            filepath: Path to the file containing identity cards
            
        Returns:
            List of loaded identity cards
        """
        # This would require implementing deserialization
        # For now, return empty list as placeholder
        logger.warning("Loading identity cards not yet implemented")
        return []
    
    def list_artifacts(self, artifact_type: str = "all") -> Dict[str, List[Path]]:
        """
        List available artifacts.
        
        Args:
            artifact_type: Type of artifacts ('cards', 'classifications', 'reports', 'all')
            
        Returns:
            Dictionary mapping artifact types to file paths
        """
        results = {}
        
        if artifact_type in ("cards", "all"):
            results["cards"] = list(self.cards_dir.glob("*.json")) + \
                              list(self.cards_dir.glob("*.yaml")) + \
                              list(self.cards_dir.glob("*.md"))
        
        if artifact_type in ("classifications", "all"):
            results["classifications"] = list(self.classifications_dir.glob("*.json")) + \
                                       list(self.classifications_dir.glob("*.yaml"))
        
        if artifact_type in ("reports", "all"):
            results["reports"] = list(self.reports_dir.glob("*"))
        
        return results
    
    def _save_cards_json(self, cards: List[IdentityCard], filepath: Path) -> None:
        """Save cards as JSON."""
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_cards': len(cards),
                'version': '1.0.0',
                'format': 'identity_cards_collection'
            },
            'identity_cards': [card.to_dict() for card in cards]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_cards_yaml(self, cards: List[IdentityCard], filepath: Path) -> None:
        """Save cards as YAML."""
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_cards': len(cards),
                'version': '1.0.0',
                'format': 'identity_cards_collection'
            },
            'identity_cards': [card.to_dict() for card in cards]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def _save_cards_markdown(self, cards: List[IdentityCard], filepath: Path) -> None:
        """Save cards as Markdown report."""
        lines = []
        lines.append(f"# Identity Cards Report")
        lines.append(f"")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Cards: {len(cards)}")
        lines.append(f"")
        
        for i, card in enumerate(cards, 1):
            lines.append(f"## {i}. {card.file_name}")
            lines.append(f"")
            lines.append(f"- **Card ID**: {card.card_id}")
            lines.append(f"- **File Type**: {card.file_type.value}")
            lines.append(f"- **Card Type**: {card.card_type.value}")
            lines.append(f"- **File Path**: `{card.file_path}`")
            lines.append(f"- **Size**: {card.file_size_bytes} bytes")
            lines.append(f"- **Confidence**: {card.classification_confidence:.1%}")
            lines.append(f"- **Complexity**: {card.complexity_level.value}")
            
            if card.detected_frameworks:
                frameworks = [f.value for f in card.detected_frameworks]
                lines.append(f"- **Frameworks**: {', '.join(frameworks)}")
            
            if card.quick_facts:
                lines.append(f"")
                lines.append(f"**Quick Facts:**")
                for fact in card.quick_facts:
                    lines.append(f"- {fact}")
            
            if card.key_insights:
                lines.append(f"")
                lines.append(f"**Key Insights:**")
                for insight in card.key_insights:
                    lines.append(f"- {insight}")
            
            if card.llm_summary:
                lines.append(f"")
                lines.append(f"**Summary**: {card.llm_summary}")
            
            lines.append(f"")
            lines.append(f"**LLM Context**: `{card.to_llm_context()}`")
            lines.append(f"")
            lines.append("---")
            lines.append(f"")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _classification_to_dict(self, classification: FileClassification) -> Dict[str, Any]:
        """Convert FileClassification to dictionary."""
        return {
            'file_path': str(classification.file_path),
            'file_type': classification.classification_result.file_type.value,
            'confidence': classification.classification_result.confidence,
            'frameworks': [f.value for f in classification.classification_result.frameworks],
            'detected_by_extension': classification.classification_result.detected_by_extension,
            'detected_by_content': classification.classification_result.detected_by_content,
            'detected_by_filename': classification.classification_result.detected_by_filename,
            'detected_by_shebang': classification.classification_result.detected_by_shebang,
            'size_bytes': classification.metrics.size_bytes,
            'line_count': classification.metrics.line_count,
            'non_empty_lines': classification.metrics.non_empty_lines,
            'complexity_level': classification.complexity_level.value if hasattr(classification, 'complexity_level') else 'unknown'
        }


class ArtifactManager:
    """
    High-level manager for working with artifacts.
    
    Provides convenience methods for common artifact operations
    and manages artifact lifecycle.
    """
    
    def __init__(self, artifacts_root: Optional[Path] = None):
        """
        Initialize artifact manager.
        
        Args:
            artifacts_root: Root directory for artifacts (defaults to ./artifacts)
        """
        if artifacts_root is None:
            artifacts_root = Path.cwd() / "artifacts"
        
        self.storage = ArtifactStorage(artifacts_root)
    
    def save_project_artifacts(
        self,
        cards: List[IdentityCard],
        classifications: List[FileClassification],
        project_name: str
    ) -> Dict[str, Path]:
        """
        Save complete project artifacts.
        
        Args:
            cards: Identity cards to save
            classifications: Classification results to save
            project_name: Name of the project
            
        Returns:
            Dictionary mapping artifact types to saved file paths
        """
        results = {}
        
        # Save identity cards in multiple formats
        results['cards_json'] = self.storage.save_identity_cards(
            cards, project_name, format="json"
        )
        results['cards_markdown'] = self.storage.save_identity_cards(
            cards, project_name, format="markdown"
        )
        
        # Save classifications
        results['classifications'] = self.storage.save_classifications(
            classifications, project_name, format="json"
        )
        
        logger.info(f"Saved complete artifacts for project '{project_name}'")
        return results
    
    def generate_summary_report(
        self,
        cards: List[IdentityCard],
        project_name: str
    ) -> Path:
        """
        Generate a summary report for the identity cards.
        
        Args:
            cards: Identity cards to summarize
            project_name: Name of the project
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_name}_summary_{timestamp}.md"
        filepath = self.storage.reports_dir / filename
        
        # Calculate summary statistics
        total_files = len(cards)
        file_types = {}
        frameworks = {}
        complexity_levels = {}
        total_size = 0
        
        for card in cards:
            # Count file types
            file_type = card.file_type.value
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Count frameworks
            for framework in card.detected_frameworks:
                fw_name = framework.value
                frameworks[fw_name] = frameworks.get(fw_name, 0) + 1
            
            # Count complexity levels
            complexity = card.complexity_level.value
            complexity_levels[complexity] = complexity_levels.get(complexity, 0) + 1
            
            # Sum file sizes
            total_size += card.file_size_bytes
        
        # Generate report
        lines = []
        lines.append(f"# Project Analysis Summary: {project_name}")
        lines.append(f"")
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append(f"**Total Files**: {total_files}")
        lines.append(f"**Total Size**: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
        lines.append(f"")
        
        # File type distribution
        lines.append(f"## File Type Distribution")
        lines.append(f"")
        for file_type, count in sorted(file_types.items()):
            percentage = (count / total_files) * 100
            lines.append(f"- **{file_type}**: {count} files ({percentage:.1f}%)")
        lines.append(f"")
        
        # Framework distribution
        if frameworks:
            lines.append(f"## Framework Distribution")
            lines.append(f"")
            for framework, count in sorted(frameworks.items()):
                lines.append(f"- **{framework}**: {count} files")
            lines.append(f"")
        
        # Complexity distribution
        lines.append(f"## Complexity Distribution")
        lines.append(f"")
        for complexity, count in sorted(complexity_levels.items()):
            percentage = (count / total_files) * 100
            lines.append(f"- **{complexity}**: {count} files ({percentage:.1f}%)")
        lines.append(f"")
        
        # Top insights
        all_insights = []
        for card in cards:
            all_insights.extend(card.key_insights)
        
        if all_insights:
            lines.append(f"## Key Insights")
            lines.append(f"")
            unique_insights = list(set(all_insights))[:10]  # Top 10 unique insights
            for insight in unique_insights:
                lines.append(f"- {insight}")
            lines.append(f"")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated summary report: {filepath}")
        return filepath