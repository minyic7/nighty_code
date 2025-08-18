# Core Scanner Module

This directory contains the core scanner functionality organized into separate modules for better maintainability.

## Module Structure

### `base.py`
Contains abstract base classes and core data structures:
- `Scanner` - Abstract base class for all scanner implementations
- `ScannerConfig` - Configuration dataclass for scanners
- `FileInfo` - Data structure for file information

### `repository.py`
Contains the repository scanner implementation:
- `RepositoryScanner` - Concrete implementation for scanning local repositories
- `scan_repository()` - Convenience function for repository scanning

### `scanner.py`
Main entry point that imports and exposes all scanner functionality:
- Imports all base classes and implementations
- Provides placeholder classes for future scanner types
- Defines `__all__` for clean public API

## Usage

### Basic Usage
```python
from nighty_code.core.scanner import RepositoryScanner, ScannerConfig

# Create scanner with default config
scanner = RepositoryScanner()
files = scanner.scan("/path/to/repo")

# Or with custom config
config = ScannerConfig(max_file_size_mb=5.0)
scanner = RepositoryScanner(config)
files = scanner.scan("/path/to/repo")
```

### Using Convenience Function
```python
from nighty_code.core.scanner import scan_repository

files = scan_repository("/path/to/repo")
```

### Direct Module Imports
```python
# Import from specific modules if needed
from nighty_code.core.base import Scanner, ScannerConfig
from nighty_code.core.repository import RepositoryScanner
```

## Future Scanner Types

The architecture is designed to support additional scanner types:

- `GitScanner` - Git-aware scanning with history
- `RemoteScanner` - GitHub/GitLab API scanning  
- `ArchiveScanner` - Scanning compressed files

These are currently placeholders that raise `NotImplementedError`.

## Extension

To add a new scanner type:

1. Create a new module (e.g., `git.py`)
2. Implement a class inheriting from `Scanner[FileInfo]`
3. Implement required abstract methods: `scan()` and `scan_iterator()`
4. Add imports to `scanner.py`
5. Add to `__all__` list

Example:
```python
from .base import Scanner, FileInfo

class MyScanner(Scanner[FileInfo]):
    def scan(self, target):
        # Implementation here
        pass
    
    def scan_iterator(self, target):
        # Implementation here  
        pass
```