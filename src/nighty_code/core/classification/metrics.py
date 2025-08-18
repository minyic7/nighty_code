"""
File metrics calculation utilities.
"""

import re
from typing import List, Pattern
from pathlib import Path

from ..models import FileType, FileMetrics
from .rules import ClassificationRules


class MetricsCalculator:
    """Calculate various metrics for file content."""
    
    def __init__(self, rules: ClassificationRules):
        self.rules = rules
    
    def calculate_metrics(self, content: str, file_type: FileType) -> FileMetrics:
        """Calculate comprehensive metrics for file content."""
        if not content:
            return FileMetrics(
                size_bytes=0,
                line_count=0,
                non_empty_lines=0,
                character_count=0,
                word_count=0
            )
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic metrics
        size_bytes = len(content.encode('utf-8'))
        line_count = len(lines)
        non_empty_line_count = len(non_empty_lines)
        character_count = len(content)
        word_count = len(content.split())
        max_line_length = max(len(line) for line in lines) if lines else 0
        
        # Language-specific metrics
        comment_lines = self._count_comment_lines(content, file_type)
        code_lines = self._count_code_lines(content, file_type, comment_lines)
        
        return FileMetrics(
            size_bytes=size_bytes,
            line_count=line_count,
            non_empty_lines=non_empty_line_count,
            comment_lines=comment_lines,
            code_lines=code_lines,
            character_count=character_count,
            word_count=word_count,
            max_line_length=max_line_length
        )
    
    def _count_comment_lines(self, content: str, file_type: FileType) -> int:
        """Count lines that are primarily comments."""
        comment_patterns = self.rules.get_comment_patterns(file_type)
        if not comment_patterns:
            return 0
        
        lines = content.split('\n')
        comment_count = 0
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # Check if line starts with comment
            for pattern in comment_patterns:
                # For single-line comments, check if line starts with comment after whitespace
                if pattern.pattern.endswith('$'):  # Single line comment pattern
                    comment_start = pattern.pattern.split('.*$')[0]
                    if stripped_line.startswith(comment_start.replace('\\', '')):
                        comment_count += 1
                        break
        
        return comment_count
    
    def _count_code_lines(self, content: str, file_type: FileType, comment_lines: int) -> int:
        """Count lines that contain actual code."""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Simple heuristic: non-empty lines minus comment lines
        # This could be enhanced with more sophisticated parsing
        code_lines = len(non_empty_lines) - comment_lines
        
        return max(0, code_lines)
    
    def calculate_complexity_score(self, content: str, file_type: FileType) -> float:
        """Calculate a complexity score for the content."""
        if not content:
            return 0.0
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Base complexity from length
        length_score = min(1.0, len(non_empty_lines) / 1000)  # Normalize to 1000 lines
        
        # Control flow complexity
        control_flow_score = self._calculate_control_flow_complexity(content, file_type)
        
        # Nesting complexity
        nesting_score = self._calculate_nesting_complexity(content, file_type)
        
        # Function/class count complexity
        structure_score = self._calculate_structure_complexity(content, file_type)
        
        # Weighted combination
        complexity = (
            length_score * 0.3 +
            control_flow_score * 0.3 +
            nesting_score * 0.2 +
            structure_score * 0.2
        )
        
        return min(1.0, complexity)
    
    def _calculate_control_flow_complexity(self, content: str, file_type: FileType) -> float:
        """Calculate complexity based on control flow statements."""
        control_patterns = {
            FileType.PYTHON: [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', r'\bexcept\b'],
            FileType.JAVA: [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', r'\bcatch\b', r'\bswitch\b'],
            FileType.SCALA: [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', r'\bcatch\b', r'\bmatch\b'],
            FileType.JAVASCRIPT: [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b', r'\bcatch\b', r'\bswitch\b'],
            FileType.SQL: [r'\bCASE\b', r'\bWHEN\b', r'\bIF\b'],
        }
        
        patterns = control_patterns.get(file_type, [])
        if not patterns:
            return 0.0
        
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            total_matches += len(matches)
        
        # Normalize based on content length
        lines = len(content.split('\n'))
        return min(1.0, total_matches / max(lines / 10, 1))
    
    def _calculate_nesting_complexity(self, content: str, file_type: FileType) -> float:
        """Calculate complexity based on nesting levels."""
        lines = content.split('\n')
        max_nesting = 0
        avg_nesting = 0
        
        if file_type in [FileType.PYTHON]:
            # Python uses indentation for nesting
            nesting_levels = []
            for line in lines:
                if line.strip():
                    indent_level = (len(line) - len(line.lstrip())) // 4  # Assume 4-space indents
                    nesting_levels.append(indent_level)
            
            if nesting_levels:
                max_nesting = max(nesting_levels)
                avg_nesting = sum(nesting_levels) / len(nesting_levels)
        
        elif file_type in [FileType.JAVA, FileType.SCALA, FileType.JAVASCRIPT]:
            # Brace-based languages
            current_nesting = 0
            nesting_levels = []
            
            for line in lines:
                current_nesting += line.count('{') - line.count('}')
                if line.strip():
                    nesting_levels.append(current_nesting)
            
            if nesting_levels:
                max_nesting = max(nesting_levels)
                avg_nesting = sum(nesting_levels) / len(nesting_levels)
        
        # Normalize to 0-1 scale
        max_score = min(1.0, max_nesting / 10)  # 10 levels = max complexity
        avg_score = min(1.0, avg_nesting / 5)   # 5 levels = high complexity
        
        return (max_score + avg_score) / 2
    
    def _calculate_structure_complexity(self, content: str, file_type: FileType) -> float:
        """Calculate complexity based on functions, classes, etc."""
        structure_patterns = {
            FileType.PYTHON: {
                'function': r'\bdef\s+\w+',
                'class': r'\bclass\s+\w+',
            },
            FileType.JAVA: {
                'method': r'\b(public|private|protected).*\w+\s*\(',
                'class': r'\b(public\s+)?class\s+\w+',
            },
            FileType.SCALA: {
                'function': r'\bdef\s+\w+',
                'class': r'\bclass\s+\w+',
                'object': r'\bobject\s+\w+',
                'trait': r'\btrait\s+\w+',
            },
            FileType.JAVASCRIPT: {
                'function': r'\bfunction\s+\w+',
                'class': r'\bclass\s+\w+',
            },
        }
        
        patterns = structure_patterns.get(file_type, {})
        if not patterns:
            return 0.0
        
        total_structures = 0
        for pattern in patterns.values():
            matches = re.findall(pattern, content, re.IGNORECASE)
            total_structures += len(matches)
        
        # Normalize based on content length
        lines = len(content.split('\n'))
        return min(1.0, total_structures / max(lines / 50, 1))  # 1 structure per 50 lines = max