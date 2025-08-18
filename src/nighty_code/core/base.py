"""
Base abstract classes for scanner implementations.
"""

import fnmatch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any, Generic, TypeVar, Callable
from dataclasses import dataclass, field
import yaml
import re


T = TypeVar('T')  # Generic type for scan results


@dataclass
class ScannerConfig:
    """Base configuration for scanners."""
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", ".venv", "venv", 
        "node_modules", ".idea", ".vscode"
    ])
    max_file_size_mb: float = 10.0
    follow_symlinks: bool = False
    scanignore_filename: str = ".scanignore"  # Custom ignore file name
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "ScannerConfig":
        """Load configuration from YAML file."""
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                scanner_config = config_data.get('scanner', {})
                # Get default values from class
                default_config = cls()
                return cls(
                    ignore_patterns=scanner_config.get('ignore_patterns', default_config.ignore_patterns),
                    max_file_size_mb=scanner_config.get('max_file_size_mb', default_config.max_file_size_mb),
                    follow_symlinks=scanner_config.get('follow_symlinks', default_config.follow_symlinks),
                    scanignore_filename=scanner_config.get('scanignore_filename', default_config.scanignore_filename)
                )
        return cls()


@dataclass
class FileInfo:
    """Information about a scanned file."""
    path: Path
    relative_path: Path
    size_bytes: int
    extension: str
    
    @property
    def size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.size_bytes / (1024 * 1024)


class Scanner(ABC, Generic[T]):
    """Abstract base class for all scanner types with built-in ignore functionality."""
    
    def __init__(self, config: Optional[ScannerConfig] = None):
        """
        Initialize the scanner.
        
        Args:
            config: Scanner configuration. If None, uses default configuration.
        """
        self.config = config or ScannerConfig()
        self._ignore_patterns: List[str] = []
        self._ignore_file_loaded = False
    
    def _load_ignore_file(self, root_path: Path, ignore_file_path: Optional[Path] = None) -> None:
        """
        Load ignore patterns from a .scanignore file.
        
        Args:
            root_path: Root directory being scanned
            ignore_file_path: Optional custom path to ignore file. If None, searches for 
                            .scanignore in root_path
        """
        if self._ignore_file_loaded:
            return
            
        ignore_file = ignore_file_path
        if ignore_file is None:
            ignore_file = root_path / self.config.scanignore_filename
        
        if ignore_file and ignore_file.exists():
            try:
                # Read and parse the ignore file patterns
                with open(ignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            self._ignore_patterns.append(line)
            except Exception:
                pass
        
        self._ignore_file_loaded = True
    
    def should_ignore(self, item: Any, root_path: Optional[Path] = None, 
                     ignore_file_path: Optional[Path] = None) -> bool:
        """
        Check if an item should be ignored based on patterns and ignore files.
        
        Args:
            item: Item to check (should be a Path)
            root_path: Root directory being scanned (for loading .scanignore)
            ignore_file_path: Optional custom path to ignore file
            
        Returns:
            True if the item should be ignored, False otherwise
        """
        if not isinstance(item, Path):
            return False
        
        # Load ignore file if we have a root path and haven't loaded it yet
        if root_path and not self._ignore_file_loaded:
            self._load_ignore_file(root_path, ignore_file_path)
        
        path = item
        path_str = str(path)
        path_name = path.name
        
        # Check against basic ignore patterns from config
        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(path_name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
        
        # Check against .scanignore file patterns if available
        for pattern in self._ignore_patterns:
            if self._matches_gitignore_pattern(pattern, path, root_path):
                return True
        
        return False
    
    def _matches_gitignore_pattern(self, pattern: str, path: Path, root_path: Optional[Path] = None) -> bool:
        """
        Check if a path matches a gitignore-style pattern.
        
        Args:
            pattern: The gitignore pattern to match against
            path: The path to check
            root_path: Optional root path for relative matching
            
        Returns:
            True if the pattern matches, False otherwise
        """
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            # This is a directory pattern - check the directory name
            dir_pattern = pattern[:-1]
            # Check if the path name matches the directory pattern
            if fnmatch.fnmatch(path.name, dir_pattern):
                return True
            # Check if any part of the path matches the directory pattern
            path_parts = str(path).replace('\\', '/').split('/')
            if dir_pattern in path_parts:
                return True
            return False
        
        # Simple filename pattern matching
        if fnmatch.fnmatch(path.name, pattern):
            return True
            
        # Full path pattern matching
        path_str = str(path).replace('\\', '/')
        if fnmatch.fnmatch(path_str, pattern):
            return True
            
        # Relative path matching if we have a root path
        if root_path:
            try:
                if path.is_absolute():
                    rel_path = str(path.relative_to(root_path)).replace('\\', '/')
                else:
                    rel_path = str(path).replace('\\', '/')
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
            except ValueError:
                pass
                
        return False
    
    def reset_ignore_state(self) -> None:
        """Reset the ignore file loading state. Useful when scanning multiple directories."""
        self._ignore_patterns = []
        self._ignore_file_loaded = False
    
    @abstractmethod
    def scan(self, target: Any) -> List[T]:
        """
        Perform a scan operation.
        
        Args:
            target: The target to scan (could be a path, URL, etc.)
            
        Returns:
            List of scan results
        """
        pass
    
    @abstractmethod
    def scan_iterator(self, target: Any) -> Iterator[T]:
        """
        Perform a scan operation with iterator.
        
        Args:
            target: The target to scan
            
        Yields:
            Scan results one by one
        """
        pass
    
    def get_statistics(self, results: List[T]) -> Dict:
        """
        Get statistics about scan results.
        Default implementation returns basic count.
        
        Args:
            results: List of scan results
            
        Returns:
            Dictionary with statistics
        """
        return {
            'total_items': len(results),
            'scanner_type': self.__class__.__name__
        }