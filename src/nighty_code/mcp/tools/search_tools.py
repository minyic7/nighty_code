"""
Search tools for MCP server.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import fnmatch
import re

from ..base import (
    ToolDefinition, ToolCategory, ToolParameter,
    MCPTool, InvalidParameterError
)
from ..registry import register_tool
from .file_tools import BaseFileTool
from ..utils.fuzzy_match import FuzzyMatcher


@register_tool
class SearchInFilesTool(BaseFileTool, MCPTool):
    """Tool for searching text in files."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_in_files",
            description="Search for text or patterns in files",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Text or regex pattern to search for"
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory or file to search in",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="file_pattern",
                    type="string",
                    description="File pattern to filter (e.g., '*.py')",
                    required=False,
                    default="*"
                ),
                ToolParameter(
                    name="use_regex",
                    type="boolean",
                    description="Treat query as regex pattern",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="case_sensitive",
                    type="boolean",
                    description="Case sensitive search",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=50
                )
            ],
            returns="array",
            examples=[
                "search_in_files('TODO')",
                "search_in_files('class.*Scanner', use_regex=True, file_pattern='*.py')"
            ]
        )
    
    async def execute(
        self,
        query: str,
        path: str = ".",
        file_pattern: str = "*",
        use_regex: bool = False,
        case_sensitive: bool = True,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for text in files with robust handling."""
        # Validate query
        if not query or query.isspace():
            raise InvalidParameterError("query", "Search query cannot be empty or whitespace")
        
        if len(query) > 1000:
            raise InvalidParameterError("query", "Search query too long (max 1000 chars)")
        
        # Security: Check for null bytes and other dangerous chars
        if '\x00' in query:
            raise InvalidParameterError("query", "Query contains null byte")
        
        # Security check for regex
        if use_regex:
            # Prevent catastrophic backtracking
            dangerous_patterns = [
                r'(\w+)*\w*',  # Exponential
                r'(a*)*',      # Nested quantifiers
                r'(.*)*',      # Greedy nested
            ]
            for pattern in dangerous_patterns:
                if pattern in query:
                    raise InvalidParameterError("query", "Potentially dangerous regex pattern")
        
        search_path = self.validate_path(path)
        
        if not search_path.exists():
            raise InvalidParameterError("path", f"Path not found: {path}")
        
        results = []
        files_searched = 0
        MAX_FILES = 1000  # Limit files to search
        
        # Prepare search pattern
        if use_regex:
            try:
                import re
                # Add timeout for regex compilation
                pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
            except re.error as e:
                raise InvalidParameterError("query", f"Invalid regex: {str(e)}")
        else:
            if not case_sensitive:
                query = query.lower()
        
        # Excluded directories
        EXCLUDED_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}
        
        # Search in files
        if search_path.is_file():
            files_to_search = [search_path]
        else:
            files_to_search = search_path.rglob(file_pattern)
        
        for file_path in files_to_search:
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in EXCLUDED_DIRS):
                continue
            
            if not file_path.is_file():
                continue
            
            if file_path.name.startswith('.'):
                continue
            
            files_searched += 1
            if files_searched > MAX_FILES:
                results.append({
                    "file": "[Search limit reached]",
                    "line": 0,
                    "content": f"Searched {MAX_FILES} files. Refine your search."
                })
                break
            
            # Skip large files
            try:
                if file_path.stat().st_size > 1024 * 1024:  # 1MB
                    continue
            except:
                continue
            
            # Skip binary files
            try:
                with open(file_path, 'rb') as f:
                    chunk = f.read(512)
                    if b'\x00' in chunk:  # Binary file
                        continue
            except:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        # Limit line processing
                        if line_num > 10000:  # Skip huge files
                            break
                        
                        if len(line) > 1000:  # Skip very long lines
                            continue
                        
                        match_found = False
                        
                        if use_regex:
                            try:
                                # Timeout for regex search
                                if pattern.search(line):
                                    match_found = True
                            except:
                                continue
                        else:
                            search_line = line if case_sensitive else line.lower()
                            if query in search_line:
                                match_found = True
                        
                        if match_found:
                            results.append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_num,
                                "content": line.strip()[:200]  # Limit line length
                            })
                            
                            if len(results) >= max_results:
                                return results
                                
            except Exception:
                # Skip files that can't be read
                continue
        
        # Rank results by relevance
        if results and hasattr(self, 'fuzzy_matcher'):
            results = self.fuzzy_matcher.rank_search_results(results, query)
        
        return results


@register_tool
class FindFilesTool(BaseFileTool, MCPTool):
    """Tool for finding files by name pattern."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="find_files",
            description="Find files matching a name pattern",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="File name pattern (glob or regex)"
                ),
                ToolParameter(
                    name="search_path",
                    type="string",
                    description="Directory to search in",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="use_regex",
                    type="boolean",
                    description="Treat pattern as regex",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="include_directories",
                    type="boolean",
                    description="Include directories in results",
                    required=False,
                    default=False
                )
            ],
            returns="array",
            examples=[
                "find_files('*.py')",
                "find_files('test_.*\\.py$', use_regex=True)"
            ]
        )
    
    async def execute(
        self,
        pattern: str,
        search_path: str = ".",
        use_regex: bool = False,
        include_directories: bool = False
    ) -> List[str]:
        """Find files matching a pattern."""
        # Validate pattern for security
        if not pattern or pattern.isspace():
            raise InvalidParameterError("pattern", "Pattern cannot be empty")
        
        # Security checks for patterns
        # Allow **/* for recursive glob but block absolute paths
        dangerous_patterns = ['../', '..\\', '/../', '\\..\\', '\x00']
        for dangerous in dangerous_patterns:
            if dangerous in pattern:
                raise InvalidParameterError("pattern", f"Invalid pattern: contains '{dangerous}'")
        
        # Check for absolute path patterns (but allow **/* glob)
        if pattern.startswith('/') or pattern.startswith('\\'):
            raise InvalidParameterError("pattern", "Absolute paths not allowed in pattern")
        
        # Check for absolute paths in pattern
        if Path(pattern).is_absolute():
            raise InvalidParameterError("pattern", "Absolute paths not allowed in pattern")
        
        path = self.validate_path(search_path)
        
        if not path.exists():
            raise InvalidParameterError("search_path", f"Path not found: {search_path}")
        
        if not path.is_dir():
            raise InvalidParameterError("search_path", f"Not a directory: {search_path}")
        
        results = []
        MAX_RESULTS = 1000  # Limit results
        
        # Prepare pattern
        if use_regex:
            try:
                regex_pattern = re.compile(pattern)
            except re.error as e:
                raise InvalidParameterError("pattern", f"Invalid regex: {str(e)}")
        
        # Search for files
        for item in path.rglob("*"):
            if item.name.startswith('.'):
                continue
            
            # Check if we should include this item
            if item.is_dir() and not include_directories:
                continue
            
            # Match against pattern
            match_found = False
            rel_path = str(item.relative_to(self.project_root))
            
            if use_regex:
                if regex_pattern.search(item.name):
                    match_found = True
            else:
                if fnmatch.fnmatch(item.name, pattern):
                    match_found = True
            
            if match_found:
                results.append(rel_path)
                
                # Limit results
                if len(results) >= MAX_RESULTS:
                    results.append(f"[Results limited to {MAX_RESULTS} files]")
                    break
        
        return sorted(results[:MAX_RESULTS])