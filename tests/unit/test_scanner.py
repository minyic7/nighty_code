"""
Unit tests for the scanner module with simplified structure.
"""

import pytest
from pathlib import Path
import tempfile
import os
from nighty_code.core import (
    Scanner,
    FolderScanner,
    GitScanner,
    ArchiveScanner,
    ScannerConfig, 
    FileInfo,
    scan_folder
)


class TestScannerConfig:
    """Tests for ScannerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScannerConfig()
        assert config.max_file_size_mb == 10.0
        assert config.follow_symlinks is False
        assert ".git" in config.ignore_patterns
        assert "__pycache__" in config.ignore_patterns
    
    def test_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
scanner:
  ignore_patterns:
    - "*.test"
    - "test_dir"
  max_file_size_mb: 5
  follow_symlinks: true
        """)
        
        config = ScannerConfig.from_yaml(config_file)
        assert config.max_file_size_mb == 5
        assert config.follow_symlinks is True
        assert "*.test" in config.ignore_patterns
        assert "test_dir" in config.ignore_patterns


class TestFileInfo:
    """Tests for FileInfo class."""
    
    def test_size_mb_property(self):
        """Test size_mb property calculation."""
        file_info = FileInfo(
            path=Path("/test/file.py"),
            relative_path=Path("file.py"),
            size_bytes=1048576,  # 1 MB
            extension=".py"
        )
        assert file_info.size_mb == 1.0


class TestAbstractScanner:
    """Tests for abstract Scanner base class."""
    
    def test_scanner_is_abstract(self):
        """Test that Scanner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Scanner()
    
    def test_scanner_inheritance(self):
        """Test that FolderScanner inherits from Scanner."""
        scanner = FolderScanner()
        assert isinstance(scanner, Scanner)
        assert isinstance(scanner, FolderScanner)


class TestFolderScanner:
    """Tests for FolderScanner implementation."""
    
    def test_initialization(self):
        """Test FolderScanner initialization."""
        scanner = FolderScanner()
        assert scanner.config is not None
        assert isinstance(scanner.config, ScannerConfig)
    
    def test_inheritance(self):
        """Test that FolderScanner inherits from Scanner."""
        scanner = FolderScanner()
        assert isinstance(scanner, Scanner)
        assert isinstance(scanner, FolderScanner)
    
    def test_should_ignore_with_config_patterns(self, tmp_path):
        """Test ignore pattern matching using config patterns."""
        scanner = FolderScanner()
        
        # Test default ignore patterns (should work without root_path)
        assert scanner.should_ignore(Path(".git"))
        assert scanner.should_ignore(Path("__pycache__"))
        assert scanner.should_ignore(Path("test.pyc"))
        assert scanner.should_ignore(Path("venv"))
        assert scanner.should_ignore(Path("node_modules"))
        
        # Test non-ignored paths
        assert not scanner.should_ignore(Path("main.py"))
        assert not scanner.should_ignore(Path("src"))
        assert not scanner.should_ignore(Path("README.md"))
    
    def test_should_ignore_with_scanignore_file(self, tmp_path):
        """Test ignore pattern matching with .scanignore file."""
        # Create .scanignore file
        scanignore = tmp_path / ".scanignore"
        scanignore.write_text("*.log\nbuild/\ntemp_*\n")
        
        scanner = FolderScanner()
        
        # Test .scanignore patterns
        assert scanner.should_ignore(Path("debug.log"), tmp_path)
        assert scanner.should_ignore(Path("build"), tmp_path)
        assert scanner.should_ignore(Path("temp_file.txt"), tmp_path)
        
        # Test non-ignored paths
        assert not scanner.should_ignore(Path("main.py"), tmp_path)
        assert not scanner.should_ignore(Path("README.md"), tmp_path)
    
    def test_scan_basic_folder(self, tmp_path):
        """Test basic folder scanning."""
        # Create test files
        (tmp_path / "file1.py").write_text("code")
        (tmp_path / "file2.txt").write_text("text")
        (tmp_path / "ignored.pyc").write_text("bytecode")  # Should be ignored
        
        scanner = FolderScanner()
        files = scanner.scan(tmp_path)
        
        assert len(files) == 2
        file_names = [f.relative_path.name for f in files]
        assert "file1.py" in file_names
        assert "file2.txt" in file_names
        assert "ignored.pyc" not in file_names
    
    def test_scan_with_scanignore(self, tmp_path):
        """Test scanning with .scanignore file."""
        # Create .scanignore
        scanignore = tmp_path / ".scanignore"
        scanignore.write_text("*.log\nbuild/\n")
        
        # Create files
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "debug.log").write_text("log")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.txt").write_text("output")
        
        scanner = FolderScanner()
        files = scanner.scan(tmp_path)
        
        file_names = [f.relative_path.name for f in files]
        assert "main.py" in file_names
        assert ".scanignore" in file_names
        assert "debug.log" not in file_names
        assert "output.txt" not in file_names
    
    def test_scan_with_custom_ignore_file(self, tmp_path):
        """Test scanning with custom ignore file path."""
        # Create custom ignore file
        custom_ignore = tmp_path / "custom.ignore"
        custom_ignore.write_text("*.temp\n")
        
        # Create files
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "file.temp").write_text("temp")
        
        scanner = FolderScanner()
        files = scanner.scan(tmp_path, ignore_file_path=custom_ignore)
        
        file_names = [f.relative_path.name for f in files]
        assert "main.py" in file_names
        assert "file.temp" not in file_names
    
    def test_is_file_too_large(self, tmp_path):
        """Test file size checking."""
        scanner = FolderScanner(
            ScannerConfig(max_file_size_mb=0.001)  # 1 KB limit
        )
        
        # Create a small file
        small_file = tmp_path / "small.txt"
        small_file.write_text("Hello")
        assert not scanner._is_file_too_large(small_file)
        
        # Create a large file
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 2000)  # > 1 KB
        assert scanner._is_file_too_large(large_file)
    
    def test_get_statistics(self):
        """Test statistics generation."""
        scanner = FolderScanner()
        
        # Test with empty list
        stats = scanner.get_statistics([])
        assert stats['total_files'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['scanner_type'] == 'FolderScanner'
        
        # Test with files
        files = [
            FileInfo(
                path=Path("/test/a.py"),
                relative_path=Path("a.py"),
                size_bytes=100,
                extension=".py"
            ),
            FileInfo(
                path=Path("/test/b.py"),
                relative_path=Path("b.py"),
                size_bytes=200,
                extension=".py"
            ),
        ]
        
        stats = scanner.get_statistics(files)
        assert stats['total_files'] == 2
        assert stats['total_size_bytes'] == 300
        assert stats['extensions']['.py']['count'] == 2
        assert stats['scanner_type'] == 'FolderScanner'


class TestPlaceholderScanners:
    """Tests for placeholder scanner implementations."""
    
    def test_git_scanner_placeholder(self):
        """Test GitScanner raises NotImplementedError."""
        scanner = GitScanner()
        with pytest.raises(NotImplementedError):
            scanner.scan("https://github.com/user/repo.git")
        with pytest.raises(NotImplementedError):
            list(scanner.scan_iterator("https://github.com/user/repo.git"))
    
    
    def test_archive_scanner_placeholder(self):
        """Test ArchiveScanner raises NotImplementedError."""
        scanner = ArchiveScanner()
        with pytest.raises(NotImplementedError):
            scanner.scan("/path/to/archive.zip")
        with pytest.raises(NotImplementedError):
            list(scanner.scan_iterator("/path/to/archive.zip"))


class TestScanFolder:
    """Tests for scan_folder function."""
    
    def test_scan_folder_function(self, tmp_path):
        """Test the convenience scan_folder function."""
        # Create some files
        (tmp_path / "file1.py").write_text("code1")
        (tmp_path / "file2.txt").write_text("text")
        (tmp_path / "ignored.pyc").write_text("bytecode")  # Should be ignored
        
        files = scan_folder(tmp_path)
        assert len(files) == 2
        
        file_names = [f.relative_path.name for f in files]
        assert "file1.py" in file_names
        assert "file2.txt" in file_names
        assert "ignored.pyc" not in file_names
    
    def test_scan_folder_with_custom_ignore_file(self, tmp_path):
        """Test scan_folder with custom ignore file."""
        # Create custom ignore file
        custom_ignore = tmp_path / "my.ignore"
        custom_ignore.write_text("*.temp\n")
        
        # Create files
        (tmp_path / "file1.py").write_text("code")
        (tmp_path / "file2.temp").write_text("temp")
        
        files = scan_folder(tmp_path, ignore_file_path=custom_ignore)
        assert len(files) == 2  # file1.py and my.ignore
        
        file_names = [f.relative_path.name for f in files]
        assert "file1.py" in file_names
        assert "my.ignore" in file_names
        assert "file2.temp" not in file_names