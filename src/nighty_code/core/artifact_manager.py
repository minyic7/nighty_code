"""
Artifact manager for handling repository artifact storage and retrieval.

This module manages the artifacts directory structure, ensuring consistent
naming and organization of repository analysis results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Import only when needed to avoid circular imports

logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    Manages artifact storage with consistent folder naming and organization.
    
    Handles:
    - Artifact directory naming based on repository name
    - Loading and saving artifacts
    - Checking artifact freshness
    - Managing multiple repository artifacts
    """
    
    def __init__(self, artifacts_root: Optional[Path] = None):
        """
        Initialize artifact manager.
        
        Args:
            artifacts_root: Root directory for all artifacts (default: ./artifacts)
        """
        if artifacts_root is None:
            artifacts_root = Path("artifacts")
        
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
    
    def get_artifacts_dir(self, repository_path: Path) -> Path:
        """
        Get artifacts directory for a repository with sanitized name.
        
        Args:
            repository_path: Path to the repository
            
        Returns:
            Path to the artifacts directory for this repository
        """
        # Sanitize repository name for use as folder name
        repo_name = self._sanitize_name(repository_path.name)
        
        # Handle special cases
        if repo_name in [".", "..", ""]:
            # Use parent directory name or fallback
            if repository_path.resolve() != Path.cwd():
                repo_name = self._sanitize_name(repository_path.resolve().name)
            else:
                repo_name = "current_repo"
        
        artifacts_dir = self.artifacts_root / repo_name
        return artifacts_dir
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize repository name for use as folder name.
        
        Args:
            name: Original repository name
            
        Returns:
            Sanitized name safe for filesystem
        """
        # Replace problematic characters
        sanitized = name.replace(".", "_")
        sanitized = sanitized.replace(" ", "_")
        sanitized = sanitized.replace("/", "_")
        sanitized = sanitized.replace("\\", "_")
        sanitized = sanitized.replace(":", "_")
        
        # Remove any remaining non-alphanumeric characters except underscore and dash
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-")
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "repo"
        
        return sanitized
    
    def ensure_artifacts_exist(self, repository_path: Path) -> bool:
        """
        Check if artifacts exist for a repository.
        
        Args:
            repository_path: Path to the repository
            
        Returns:
            True if core artifacts exist
        """
        artifacts_dir = self.get_artifacts_dir(repository_path)
        
        # Check for core artifact files
        required_files = [
            "identity_cards_all.json",
            "relationships_all.json",
            "repository_graph.json"
        ]
        
        return all((artifacts_dir / f).exists() for f in required_files)
    
    def get_artifact_storage(self, repository_path: Path):
        """
        Get an ArtifactStorage instance for a repository.
        
        Args:
            repository_path: Path to the repository
            
        Returns:
            Configured ArtifactStorage instance
        """
        # Import here to avoid circular dependency
        from ..storage.artifacts import ArtifactStorage
        
        artifacts_dir = self.get_artifacts_dir(repository_path)
        return ArtifactStorage(artifacts_dir, create_dirs=True)
    
    def load_artifacts(self, repository_path: Path) -> Dict[str, Any]:
        """
        Load all artifacts for a repository.
        
        Args:
            repository_path: Path to the repository
            
        Returns:
            Dictionary containing all artifacts
        """
        artifacts_dir = self.get_artifacts_dir(repository_path)
        artifacts = {}
        
        # Load each artifact type
        artifact_files = {
            "identity_cards": "identity_cards_all.json",
            "entities": "entities_all.json",
            "relationships": "relationships_all.json",
            "repository_graph": "repository_graph.json",
            "classifications": "classifications_all_files.json"
        }
        
        for key, filename in artifact_files.items():
            filepath = artifacts_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        artifacts[key] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
                    artifacts[key] = None
            else:
                artifacts[key] = None
        
        return artifacts
    
    def get_artifact_metadata(self, repository_path: Path) -> Dict[str, Any]:
        """
        Get metadata about artifacts for a repository.
        
        Args:
            repository_path: Path to the repository
            
        Returns:
            Metadata dictionary with creation time, size, etc.
        """
        artifacts_dir = self.get_artifacts_dir(repository_path)
        
        if not artifacts_dir.exists():
            return {
                "exists": False,
                "path": str(artifacts_dir)
            }
        
        metadata = {
            "exists": True,
            "path": str(artifacts_dir),
            "files": {}
        }
        
        # Get info for each artifact file
        for filepath in artifacts_dir.glob("*.json"):
            stat = filepath.stat()
            metadata["files"][filepath.name] = {
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        
        return metadata
    
    def list_repositories(self) -> List[str]:
        """
        List all repositories with artifacts.
        
        Returns:
            List of repository names with artifacts
        """
        repos = []
        
        if self.artifacts_root.exists():
            for item in self.artifacts_root.iterdir():
                if item.is_dir():
                    # Check if it has any artifacts
                    if any(item.glob("*.json")):
                        repos.append(item.name)
        
        return sorted(repos)
    
    def clean_old_artifacts(self, days: int = 30) -> int:
        """
        Remove artifacts older than specified days.
        
        Args:
            days: Number of days to keep artifacts
            
        Returns:
            Number of artifacts removed
        """
        import time
        
        removed = 0
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for repo_dir in self.artifacts_root.iterdir():
            if repo_dir.is_dir():
                # Check modification time of artifacts
                old_files = []
                for filepath in repo_dir.glob("*.json"):
                    if filepath.stat().st_mtime < cutoff_time:
                        old_files.append(filepath)
                
                # Remove old artifacts
                if old_files:
                    for filepath in old_files:
                        filepath.unlink()
                        removed += 1
                    
                    # Remove directory if empty
                    if not any(repo_dir.glob("*")):
                        repo_dir.rmdir()
        
        logger.info(f"Removed {removed} old artifact files")
        return removed