"""
Test module structure and imports.
"""

import pytest


def test_scanner_module_imports():
    """Test that all scanner components can be imported from main module."""
    from nighty_code.core import (
        Scanner,
        ScannerConfig, 
        FileInfo,
        FolderScanner,
        GitScanner,
        ArchiveScanner,
        scan_folder
    )
    
    # Test that classes are available
    assert Scanner is not None
    assert ScannerConfig is not None
    assert FileInfo is not None
    assert FolderScanner is not None
    assert GitScanner is not None
    assert ArchiveScanner is not None
    assert scan_folder is not None


def test_base_module_imports():
    """Test that base components can be imported directly."""
    from nighty_code.core.base import (
        Scanner,
        ScannerConfig,
        FileInfo
    )
    
    assert Scanner is not None
    assert ScannerConfig is not None
    assert FileInfo is not None


def test_folder_scanner_module_imports():
    """Test that folder scanner can be imported directly."""
    from nighty_code.core.scanner.folder_scanner import (
        FolderScanner,
        scan_folder
    )
    
    assert FolderScanner is not None
    assert scan_folder is not None


def test_scanner_inheritance():
    """Test that FolderScanner properly inherits from Scanner."""
    from nighty_code.core import Scanner, FolderScanner
    
    scanner = FolderScanner()
    assert isinstance(scanner, Scanner)
    assert isinstance(scanner, FolderScanner)


def test_module_all_exports():
    """Test that __all__ is properly defined."""
    from nighty_code import core
    
    # Check that __all__ exists and contains expected items
    assert hasattr(core, '__all__')
    all_items = core.__all__
    
    expected_items = [
        'Scanner', 'ScannerConfig', 'FileInfo',
        'FolderScanner', 'GitScanner', 'ArchiveScanner',
        'scan_folder'
    ]
    
    for item in expected_items:
        assert item in all_items, f"{item} not found in __all__"