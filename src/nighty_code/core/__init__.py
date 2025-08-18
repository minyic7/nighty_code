"""
Core module for nighty_code repository analysis.
"""

# Import scanner functionality
from .scanner import (
    Scanner,
    ScannerConfig,
    FileInfo,
    FolderScanner,
    GitScanner,
    ArchiveScanner,
    scan_folder,
)

__all__ = [
    # Scanner functionality
    'Scanner',
    'ScannerConfig', 
    'FileInfo',
    'FolderScanner',
    'GitScanner',
    'ArchiveScanner',
    'scan_folder',
]