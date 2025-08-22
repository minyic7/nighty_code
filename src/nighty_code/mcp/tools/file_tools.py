"""
File operation tools for MCP server.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import fnmatch

from ..base import (
    ToolDefinition, ToolCategory, ToolParameter,
    MCPTool, InvalidParameterError, SecurityError
)
from ..registry import register_tool
from ..utils.fuzzy_match import FuzzyMatcher


class BaseFileTool:
    """Base class for file-based tools."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.fuzzy_matcher = FuzzyMatcher(self.project_root)
    
    def validate_path(self, file_path: str) -> Path:
        """Validate and resolve a file path with robust normalization."""
        try:
            # Strip whitespace
            file_path = file_path.strip()
            
            # Security: Reject URLs and command injection attempts
            dangerous_patterns = [
                'http://', 'https://', 'file://', 'ftp://',
                ';', '`', '$', '%', '|', '&', '\n', '\r',
                '\x00'  # Null byte
            ]
            for pattern in dangerous_patterns:
                if pattern in file_path:
                    raise SecurityError(f"Invalid characters in path: {pattern}")
            
            # Normalize path separators
            file_path = file_path.replace('\\', '/')
            
            # Remove redundant slashes and dots
            while '//' in file_path:
                file_path = file_path.replace('//', '/')
            file_path = file_path.replace('/./', '/')
            
            # Remove trailing slash from file paths
            if file_path.endswith('/'):
                file_path = file_path.rstrip('/')
            
            # Use Path for proper resolution
            path = Path(file_path)
            
            # Convert to absolute path relative to project root
            if path.is_absolute():
                # Reject absolute paths
                raise SecurityError("Absolute paths not allowed")
            
            # Resolve relative to project root
            resolved = (self.project_root / path).resolve()
            
            # Security check: ensure path is within project root
            try:
                resolved.relative_to(self.project_root)
            except ValueError:
                raise SecurityError(f"Access denied: path outside project root")
            
            return resolved
            
        except SecurityError:
            raise
        except Exception as e:
            raise InvalidParameterError("file_path", str(e))


@register_tool
class ReadFileTool(BaseFileTool, MCPTool):
    """Tool for reading file contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a file",
            category=ToolCategory.FILE,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to read"
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="Starting line number (1-based)",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="end_line",
                    type="integer", 
                    description="Ending line number (inclusive)",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8"
                )
            ],
            returns="string",
            examples=[
                "read_file('src/main.py')",
                "read_file('src/main.py', start_line=10, end_line=20)"
            ]
        )
    
    async def execute(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        encoding: str = "utf-8"
    ) -> str:
        """Read file contents with robust handling."""
        path = self.validate_path(file_path)
        
        if not path.exists():
            # Try to suggest similar files
            suggestions = self.fuzzy_matcher.suggest_corrections(file_path)
            if suggestions:
                suggestion_text = f"File not found: {file_path}. Did you mean: {', '.join(suggestions[:3])}?"
            else:
                suggestion_text = f"File not found: {file_path}"
            raise InvalidParameterError("file_path", suggestion_text)
        
        if not path.is_file():
            if path.is_dir():
                raise InvalidParameterError("file_path", f"Path is a directory, not a file: {file_path}")
            raise InvalidParameterError("file_path", f"Not a regular file: {file_path}")
        
        # Check file size to prevent memory issues
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise InvalidParameterError(
                "file_path", 
                f"File too large ({file_size / 1024 / 1024:.1f}MB). Max size: 10MB"
            )
        
        # Check if file is likely binary
        def is_binary(file_path: Path, check_bytes: int = 512) -> bool:
            with open(file_path, 'rb') as f:
                chunk = f.read(check_bytes)
                if b'\x00' in chunk:  # Null bytes indicate binary
                    return True
                # Check for high ratio of non-text bytes
                text_chars = bytes(range(32, 127)) + b'\n\r\t\f\b'
                non_text = sum(1 for byte in chunk if byte not in text_chars)
                return non_text / len(chunk) > 0.3 if chunk else False
        
        try:
            if is_binary(path):
                return f"[Binary file - {file_size} bytes]"
        except:
            pass  # Continue with text reading
        
        try:
            # Handle line range parameters
            if start_line is not None or end_line is not None:
                with open(path, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                    
                    # Normalize line numbers
                    total_lines = len(lines)
                    
                    # Handle negative or zero line numbers
                    if start_line is not None:
                        start = max(0, (start_line - 1) if start_line > 0 else 0)
                    else:
                        start = 0
                    
                    if end_line is not None:
                        if end_line < 0:
                            # Negative means from end
                            end = total_lines + end_line + 1
                        else:
                            end = min(total_lines, end_line)
                    else:
                        end = total_lines
                    
                    # Swap if start > end
                    if start > end:
                        start, end = end, start
                    
                    # Limit range to prevent huge reads
                    MAX_LINES = 10000
                    if end - start > MAX_LINES:
                        end = start + MAX_LINES
                        lines_content = ''.join(lines[start:end])
                        return f"{lines_content}\n[Truncated to {MAX_LINES} lines]"
                    
                    return ''.join(lines[start:end])
            else:
                # Read full file with size limit
                with open(path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read(MAX_FILE_SIZE)
                    if file_size > MAX_FILE_SIZE:
                        content += f"\n[File truncated - showing first {MAX_FILE_SIZE} bytes of {file_size}]"
                    return content
                    
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(path, 'r', encoding='latin-1', errors='replace') as f:
                    return f.read(MAX_FILE_SIZE)
            except Exception:
                return f"[Unable to decode file - likely binary or unknown encoding]"
        except Exception as e:
            raise InvalidParameterError("file_path", f"Error reading file: {str(e)}")


@register_tool
class ListDirectoryTool(BaseFileTool, MCPTool):
    """Tool for listing directory contents."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            category=ToolCategory.FILE,
            parameters=[
                ToolParameter(
                    name="directory_path",
                    type="string",
                    description="Path to the directory",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="File pattern to filter (e.g., '*.py')",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="List recursively",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="include_hidden",
                    type="boolean",
                    description="Include hidden files/directories",
                    required=False,
                    default=False
                )
            ],
            returns="object",
            examples=[
                "list_directory('src')",
                "list_directory('src', pattern='*.py', recursive=True)"
            ]
        )
    
    async def execute(
        self,
        directory_path: str = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
        include_hidden: bool = False
    ) -> Dict[str, Any]:
        """List directory contents."""
        path = self.validate_path(directory_path)
        
        if not path.exists():
            raise InvalidParameterError("directory_path", f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise InvalidParameterError("directory_path", f"Not a directory: {directory_path}")
        
        files = []
        directories = []
        
        try:
            if recursive:
                # Recursive listing
                for item in path.rglob(pattern or "*"):
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    rel_path = str(item.relative_to(self.project_root))
                    if item.is_file():
                        files.append(rel_path)
                    elif item.is_dir():
                        directories.append(rel_path)
            else:
                # Non-recursive listing
                for item in path.iterdir():
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    if pattern and not fnmatch.fnmatch(item.name, pattern):
                        continue
                    
                    rel_path = str(item.relative_to(self.project_root))
                    if item.is_file():
                        files.append(rel_path)
                    elif item.is_dir():
                        directories.append(rel_path)
            
            return {
                "path": str(path.relative_to(self.project_root)),
                "files": sorted(files),
                "directories": sorted(directories),
                "total_files": len(files),
                "total_directories": len(directories)
            }
            
        except Exception as e:
            raise InvalidParameterError("directory_path", f"Error listing directory: {str(e)}")