# src/mcp/servers/filesystem.py
"""Filesystem MCP server for file operations"""

import os
import aiofiles
import glob
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..core.base_server import BaseMCPServer, ToolDefinition
from ..core.types import TextContent, ErrorContent
from ..core.exceptions import MCPValidationError


logger = logging.getLogger(__name__)


class FilesystemServer(BaseMCPServer):
    """
    MCP server for filesystem operations.
    
    Provides tools for:
    - Reading files
    - Listing directories
    - Pattern search (glob)
    - Keyword search in files
    - Context-aware search (future)
    """
    
    def __init__(self, 
                 root_path: str = None,
                 allowed_extensions: List[str] = None,
                 max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__("filesystem-server", "1.0.0")
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.allowed_extensions = allowed_extensions or []
        self.max_file_size = max_file_size
        
        # Validate root path
        if not self.root_path.exists():
            raise MCPValidationError(f"Root path does not exist: {self.root_path}")
            
        logger.info(f"Filesystem server initialized with root: {self.root_path}")
        
    async def _register_tools(self):
        """Register filesystem tools"""
        
        # Read file tool
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to root"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8"
                    }
                },
                "required": ["path"]
            },
            handler=self._read_file
        ))
        
        # List directory tool
        self.register_tool(ToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to root",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                        "default": False
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files",
                        "default": False
                    }
                },
                "required": []
            },
            handler=self._list_directory
        ))
        
        # Pattern search tool
        self.register_tool(ToolDefinition(
            name="search_pattern",
            description="Search for files matching a glob pattern",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Starting directory (default: root)",
                        "default": "."
                    }
                },
                "required": ["pattern"]
            },
            handler=self._search_pattern
        ))
        
        # Keyword search tool
        self.register_tool(ToolDefinition(
            name="search_keyword",
            description="Search for a keyword in files",
            input_schema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword or regex pattern to search"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": "."
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to search (e.g., ['.py', '.js'])",
                        "default": []
                    },
                    "use_regex": {
                        "type": "boolean",
                        "description": "Treat keyword as regex",
                        "default": False
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 100
                    }
                },
                "required": ["keyword"]
            },
            handler=self._search_keyword
        ))
        
    def _validate_path(self, path: str) -> Path:
        """Validate and resolve path within root"""
        # Convert to Path object
        requested_path = Path(path)
        
        # Resolve relative to root
        if requested_path.is_absolute():
            full_path = requested_path
        else:
            full_path = self.root_path / requested_path
            
        # Resolve to real path (following symlinks)
        try:
            full_path = full_path.resolve()
        except Exception as e:
            raise MCPValidationError(f"Invalid path: {e}")
            
        # Ensure path is within root
        if not str(full_path).startswith(str(self.root_path)):
            raise MCPValidationError(f"Path outside root directory: {path}")
            
        return full_path
        
    async def _read_file(self, path: str, encoding: str = "utf-8") -> TextContent:
        """Read file contents"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.is_file():
                raise MCPValidationError(f"Not a file: {path}")
                
            # Check file size
            file_size = full_path.stat().st_size
            if file_size > self.max_file_size:
                raise MCPValidationError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
                
            # Check extension if restrictions are set
            if self.allowed_extensions:
                if full_path.suffix not in self.allowed_extensions:
                    raise MCPValidationError(f"File extension not allowed: {full_path.suffix}")
                    
            # Read file
            async with aiofiles.open(full_path, 'r', encoding=encoding) as f:
                content = await f.read()
                
            return TextContent(text=content)
            
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise
            
    async def _list_directory(self, 
                            path: str = ".",
                            recursive: bool = False,
                            include_hidden: bool = False) -> TextContent:
        """List directory contents"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.is_dir():
                raise MCPValidationError(f"Not a directory: {path}")
                
            entries = []
            
            if recursive:
                # Recursive listing
                for root, dirs, files in os.walk(full_path):
                    # Filter hidden if needed
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        files = [f for f in files if not f.startswith('.')]
                        
                    rel_root = Path(root).relative_to(self.root_path)
                    
                    for d in dirs:
                        entries.append(f"[DIR]  {rel_root / d}")
                    for f in files:
                        file_path = Path(root) / f
                        size = file_path.stat().st_size
                        entries.append(f"[FILE] {rel_root / f} ({size} bytes)")
            else:
                # Non-recursive listing
                for item in full_path.iterdir():
                    if not include_hidden and item.name.startswith('.'):
                        continue
                        
                    rel_path = item.relative_to(self.root_path)
                    if item.is_dir():
                        entries.append(f"[DIR]  {rel_path}")
                    else:
                        size = item.stat().st_size
                        entries.append(f"[FILE] {rel_path} ({size} bytes)")
                        
            entries.sort()
            result = f"Directory listing for: {path}\n" + "\n".join(entries)
            return TextContent(text=result)
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            raise
            
    async def _search_pattern(self, pattern: str, path: str = ".") -> TextContent:
        """Search for files matching pattern"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.is_dir():
                raise MCPValidationError(f"Not a directory: {path}")
                
            # Use glob to find matches
            search_path = full_path / pattern
            matches = list(full_path.glob(pattern))
            
            # Filter to files only and make relative
            results = []
            for match in matches:
                if match.is_file():
                    rel_path = match.relative_to(self.root_path)
                    size = match.stat().st_size
                    results.append(f"{rel_path} ({size} bytes)")
                    
            result = f"Found {len(results)} files matching '{pattern}':\n" + "\n".join(results)
            return TextContent(text=result)
            
        except Exception as e:
            logger.error(f"Error searching pattern {pattern}: {e}")
            raise
            
    async def _search_keyword(self,
                             keyword: str,
                             path: str = ".",
                             extensions: List[str] = None,
                             use_regex: bool = False,
                             max_results: int = 100) -> TextContent:
        """Search for keyword in files"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.is_dir():
                raise MCPValidationError(f"Not a directory: {path}")
                
            # Compile regex if needed
            if use_regex:
                pattern = re.compile(keyword)
            else:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                
            results = []
            files_searched = 0
            
            # Search files
            for file_path in full_path.rglob("*"):
                if not file_path.is_file():
                    continue
                    
                # Check extension filter
                if extensions and file_path.suffix not in extensions:
                    continue
                    
                # Skip large files
                if file_path.stat().st_size > self.max_file_size:
                    continue
                    
                files_searched += 1
                
                try:
                    # Search in file
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                        
                    matches = list(pattern.finditer(content))
                    if matches:
                        rel_path = file_path.relative_to(self.root_path)
                        
                        # Get line numbers and context
                        lines = content.splitlines()
                        for match in matches[:5]:  # Limit matches per file
                            line_num = content[:match.start()].count('\n') + 1
                            line_text = lines[line_num - 1].strip()[:100]  # Truncate long lines
                            
                            results.append(f"{rel_path}:{line_num}: {line_text}")
                            
                            if len(results) >= max_results:
                                break
                                
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    # Skip files that can't be read
                    logger.debug(f"Skipping file {file_path}: {e}")
                    continue
                    
            result = f"Search results for '{keyword}' (searched {files_searched} files):\n"
            result += "\n".join(results) if results else "No matches found"
            
            return TextContent(text=result)
            
        except Exception as e:
            logger.error(f"Error searching keyword {keyword}: {e}")
            raise
            
    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool by name (fallback for base class)"""
        # This is called if tool doesn't have a handler
        # Should not happen with our implementation
        raise NotImplementedError(f"Tool {name} not implemented")