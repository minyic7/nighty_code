"""
Folder scanner implementation for traversing local directories and collecting files.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any

from ..base import Scanner, ScannerConfig, FileInfo


class FolderScanner(Scanner[FileInfo]):
    """
    Scanner for traversing local folder directories and collecting files.
    
    This scanner handles all types of local folders and automatically
    detects and respects .scanignore files if present.
    """
    
    def _is_file_too_large(self, file_path: Path) -> bool:
        """
        Check if a file exceeds the maximum size limit.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is too large, False otherwise
        """
        try:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            return size_mb > self.config.max_file_size_mb
        except (OSError, IOError):
            return True
    
    def scan_iterator(self, target: Any, ignore_file_path: Optional[Path] = None) -> Iterator[FileInfo]:
        """
        Scan a directory and yield information about each file.
        
        Args:
            target: Root directory path to scan
            ignore_file_path: Optional custom path to ignore file
            
        Yields:
            FileInfo objects for each valid file found
        """
        root_path = Path(target).resolve()
        
        # Reset ignore state for fresh scan
        self.reset_ignore_state()
        
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(
            root_path, 
            followlinks=self.config.follow_symlinks
        ):
            current_dir = Path(dirpath)
            
            # Filter out ignored directories
            dirnames[:] = [
                d for d in dirnames 
                if not self.should_ignore(current_dir / d, root_path, ignore_file_path)
            ]
            
            # Process files
            for filename in filenames:
                file_path = current_dir / filename
                
                # Skip ignored files
                if self.should_ignore(file_path, root_path, ignore_file_path):
                    continue
                
                # Skip files that are too large
                if self._is_file_too_large(file_path):
                    continue
                
                # Get file info
                try:
                    stat = file_path.stat()
                    relative_path = file_path.relative_to(root_path)
                    
                    yield FileInfo(
                        path=file_path,
                        relative_path=relative_path,
                        size_bytes=stat.st_size,
                        extension=file_path.suffix.lower()
                    )
                except (OSError, IOError):
                    # Skip files we can't read
                    continue
    
    def scan(self, target: Any, ignore_file_path: Optional[Path] = None) -> List[FileInfo]:
        """
        Scan a directory and return a list of all files.
        
        Args:
            target: Root directory path to scan
            ignore_file_path: Optional custom path to ignore file
            
        Returns:
            List of FileInfo objects
        """
        return list(self.scan_iterator(target, ignore_file_path))
    
    # Backward compatibility methods
    def scan_directory(self, root_path: Path, ignore_file_path: Optional[Path] = None) -> Iterator[FileInfo]:
        """
        Scan a directory and yield information about each file.
        Kept for backward compatibility.
        
        Args:
            root_path: Root directory to scan
            ignore_file_path: Optional custom path to ignore file
            
        Yields:
            FileInfo objects for each valid file found
        """
        return self.scan_iterator(root_path, ignore_file_path)
    
    def get_statistics(self, results: List[FileInfo]) -> Dict:
        """
        Get statistics about scanned files.
        
        Args:
            results: List of FileInfo objects
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'extensions': {},
                'largest_file': None,
                'scanner_type': self.__class__.__name__
            }
        
        total_size = sum(f.size_bytes for f in results)
        extensions = {}
        for f in results:
            ext = f.extension or 'no_extension'
            if ext not in extensions:
                extensions[ext] = {'count': 0, 'size_bytes': 0}
            extensions[ext]['count'] += 1
            extensions[ext]['size_bytes'] += f.size_bytes
        
        largest_file = max(results, key=lambda f: f.size_bytes)
        
        return {
            'total_files': len(results),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'extensions': extensions,
            'largest_file': {
                'path': str(largest_file.relative_path),
                'size_bytes': largest_file.size_bytes,
                'size_mb': largest_file.size_mb
            },
            'scanner_type': self.__class__.__name__
        }


# Convenience function 
def scan_folder(
    root_path: Path, 
    config_path: Optional[Path] = None,
    ignore_file_path: Optional[Path] = None
) -> List[FileInfo]:
    """
    Convenience function to scan a local folder.
    
    Args:
        root_path: Root directory to scan
        config_path: Optional path to configuration file
        ignore_file_path: Optional custom path to ignore file
        
    Returns:
        List of FileInfo objects
    """
    if config_path:
        config = ScannerConfig.from_yaml(config_path)
    else:
        # Try to load from default location
        default_config = Path(root_path) / 'config' / 'default.yaml'
        if default_config.exists():
            config = ScannerConfig.from_yaml(default_config)
        else:
            config = ScannerConfig()
    
    scanner = FolderScanner(config)
    return scanner.scan(Path(root_path), ignore_file_path)