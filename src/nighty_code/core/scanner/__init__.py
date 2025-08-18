"""
Scanner package for repository analysis.

This package provides scanning functionality based on usage patterns:
- FolderScanner: For local directories (any type of folder)
- GitScanner: For remote Git repository URLs  
- ArchiveScanner: For compressed archive files
"""

# Import base classes and utilities
from ..base import (
    Scanner,
    ScannerConfig,
    FileInfo
)

# Import concrete scanner implementations
from .folder_scanner import FolderScanner, scan_folder

# Import placeholder scanner implementations
from .git_scanner import GitScanner
from .archive_scanner import ArchiveScanner

__all__ = [
    # Base classes
    'Scanner',
    'ScannerConfig', 
    'FileInfo',
    
    # Scanner implementations
    'FolderScanner',
    'GitScanner',
    'ArchiveScanner',
    
    # Convenience functions
    'scan_folder',
]