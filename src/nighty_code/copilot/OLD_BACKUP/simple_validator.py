"""
Simple input validation.
Separate concern, focused responsibility.
"""

import re
from pathlib import Path
from typing import Optional


class SimpleInputValidator:
    """
    Simple input validator.
    Just validates, nothing else.
    """
    
    def __init__(
        self,
        max_query_length: int = 1000,
        max_path_depth: int = 10,
        allowed_extensions: Optional[set] = None
    ):
        self.max_query_length = max_query_length
        self.max_path_depth = max_path_depth
        self.allowed_extensions = allowed_extensions or {
            'py', 'js', 'ts', 'java', 'go', 'rs', 'cpp', 'c', 'h',
            'json', 'yaml', 'yml', 'xml', 'md', 'txt', 'toml'
        }
        
        # Dangerous patterns to block
        self.dangerous_patterns = [
            re.compile(r'\.\.[\\/]'),  # Path traversal
            re.compile(r'[;&|`$]'),     # Command injection
            re.compile(r'<script', re.I),  # XSS
            re.compile(r"'\s*OR\s*'", re.I),  # SQL injection
        ]
    
    def validate_query(self, query: str) -> str:
        """
        Validate and clean query.
        Returns cleaned query or raises ValueError.
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        if len(query) > self.max_query_length:
            raise ValueError(f"Query too long: {len(query)} > {self.max_query_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(query):
                raise ValueError("Query contains potentially dangerous content")
        
        # Clean up whitespace
        return ' '.join(query.split())
    
    def validate_path(self, path: str) -> str:
        """
        Validate file path.
        Returns cleaned path or raises ValueError.
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Check for null bytes
        if '\x00' in path:
            raise ValueError("Path contains null bytes")
        
        # Parse path
        path_obj = Path(path)
        
        # Check for path traversal
        if '..' in path_obj.parts:
            raise ValueError("Path traversal detected")
        
        # Check depth
        if len(path_obj.parts) > self.max_path_depth:
            raise ValueError(f"Path too deep: {len(path_obj.parts)} > {self.max_path_depth}")
        
        # Check extension if it's a file
        if '.' in path_obj.name:
            ext = path_obj.suffix[1:].lower()  # Remove the dot
            if ext not in self.allowed_extensions:
                raise ValueError(f"File extension not allowed: {ext}")
        
        return str(path_obj)