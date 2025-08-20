"""
Core module for nighty_code repository analysis.
"""

# Import core functionality
from .classifier import FileClassifier
from .repository_context import RepositoryContext
from .artifact_manager import ArtifactManager

# Import scanner functionality
from .scanner import (
    Scanner,
    ScannerConfig,
    FileInfo,
    FolderScanner,
    scan_folder,
)

__all__ = [
    # Core functionality
    'FileClassifier',
    'RepositoryContext',
    'ArtifactManager',
    # Scanner functionality
    'Scanner',
    'ScannerConfig', 
    'FileInfo',
    'FolderScanner',
    'scan_folder',
]