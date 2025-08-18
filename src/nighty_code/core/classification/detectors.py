"""
File type detection implementations using various strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from pathlib import Path
import re

from ..models import FileType, Framework, ClassificationResult
from .rules import ClassificationRules


class BaseDetector(ABC):
    """Abstract base class for file type detectors."""
    
    def __init__(self, rules: ClassificationRules):
        self.rules = rules
    
    @abstractmethod
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """
        Detect file type and return confidence score.
        
        Args:
            file_path: Path to the file
            content: Optional file content
            
        Returns:
            Tuple of (FileType, confidence) or None if no detection
        """
        pass


class ExtensionDetector(BaseDetector):
    """Detect file type based on file extension."""
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Detect file type by extension with high confidence."""
        extension = file_path.suffix.lower()
        
        if extension in self.rules.extension_mapping:
            file_type = self.rules.extension_mapping[extension]
            # Extension-based detection has high confidence
            confidence = 0.9
            return (file_type, confidence)
        
        return None


class FilenameDetector(BaseDetector):
    """Detect file type based on filename patterns."""
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Detect file type by filename patterns."""
        filename = str(file_path).replace('\\', '/')
        
        for pattern, file_type in self.rules.filename_patterns.items():
            if pattern.search(filename):
                # Filename pattern detection has medium-high confidence
                confidence = 0.8
                return (file_type, confidence)
        
        return None


class ShebangDetector(BaseDetector):
    """Detect file type based on shebang lines."""
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Detect file type by shebang line."""
        if not content:
            return None
        
        # Check first line for shebang
        first_line = content.split('\n')[0].strip()
        
        for pattern, file_type in self.rules.shebang_patterns.items():
            if pattern.search(first_line):
                # Shebang detection has very high confidence
                confidence = 0.95
                return (file_type, confidence)
        
        return None


class ContentDetector(BaseDetector):
    """Detect file type based on content analysis."""
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Detect file type by analyzing content patterns."""
        if not content:
            return None
        
        # Score each file type based on pattern matches
        type_scores = {}
        
        for file_type, patterns in self.rules.content_patterns.items():
            score = self._calculate_pattern_score(content, patterns)
            if score > 0:
                type_scores[file_type] = score
        
        if not type_scores:
            return None
        
        # Get the file type with highest score
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Convert score to confidence (0.0 to 1.0)
        # Normalize based on number of patterns and content length
        max_possible_score = len(self.rules.content_patterns[best_type])
        confidence = min(0.8, best_score / max_possible_score)  # Cap at 0.8 for content detection
        
        if confidence > 0.3:  # Minimum threshold
            return (best_type, confidence)
        
        return None
    
    def _calculate_pattern_score(self, content: str, patterns: List[re.Pattern]) -> float:
        """Calculate score based on pattern matches."""
        score = 0.0
        content_lower = content.lower()
        
        for pattern in patterns:
            matches = pattern.findall(content_lower)
            if matches:
                # Give points for matches, with diminishing returns
                match_count = len(matches)
                score += min(1.0, match_count * 0.3)
        
        return score


class FrameworkDetector(BaseDetector):
    """Detect frameworks and libraries used in the file."""
    
    def detect_frameworks(self, content: str) -> List[Framework]:
        """Detect all frameworks present in the content."""
        detected_frameworks = []
        
        if not content:
            return detected_frameworks
        
        content_lower = content.lower()
        
        for framework, patterns in self.rules.framework_patterns.items():
            for pattern in patterns:
                if pattern.search(content_lower):
                    detected_frameworks.append(framework)
                    break  # One match per framework is enough
        
        return detected_frameworks
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Framework detector doesn't determine file type directly."""
        return None


class HeuristicDetector(BaseDetector):
    """Advanced heuristic-based detection for edge cases."""
    
    def detect(self, file_path: Path, content: Optional[str] = None) -> Optional[Tuple[FileType, float]]:
        """Use heuristics for complex detection scenarios."""
        if not content:
            return None
        
        # Detect binary vs text files
        if self._is_binary_content(content):
            return (FileType.BINARY, 0.9)
        
        # Detect specific patterns that are hard to catch otherwise
        
        # DBT files (SQL with DBT macros)
        if self._is_dbt_file(content):
            return (FileType.DBT, 0.85)
        
        # Spark SQL (SQL with Spark-specific functions)
        if self._is_spark_sql(content):
            return (FileType.SPARK_SQL, 0.8)
        
        # Configuration files that look like text
        if self._is_config_file(file_path, content):
            return self._detect_config_type(content)
        
        # Fallback to generic text if it's readable text
        if self._is_readable_text(content):
            return (FileType.TEXT, 0.5)
        
        return None
    
    def _is_binary_content(self, content: str) -> bool:
        """Check if content appears to be binary."""
        # Simple heuristic: check for null bytes and non-printable characters
        null_count = content.count('\x00')
        if null_count > 0:
            return True
        
        # Check for high ratio of non-printable characters
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        if len(content) > 0:
            printable_ratio = printable_chars / len(content)
            return printable_ratio < 0.7
        
        return False
    
    def _is_dbt_file(self, content: str) -> bool:
        """Check if content appears to be a DBT file."""
        dbt_patterns = [
            r'\{\{\s*config\s*\(',
            r'\{\{\s*ref\s*\(',
            r'\{\{\s*source\s*\(',
            r'\{\{\s*var\s*\(',
        ]
        
        for pattern in dbt_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _is_spark_sql(self, content: str) -> bool:
        """Check if content appears to be Spark SQL."""
        spark_patterns = [
            r'\bDELTA\b',
            r'\bPARQUET\b',
            r'\bUSING\s+DELTA\b',
            r'\bCREATE\s+TABLE\s+.*\s+USING\b',
        ]
        
        sql_patterns = [
            r'\bSELECT\b',
            r'\bFROM\b',
            r'\bWHERE\b',
        ]
        
        # Must have SQL patterns AND Spark-specific patterns
        has_sql = any(re.search(p, content, re.IGNORECASE) for p in sql_patterns)
        has_spark = any(re.search(p, content, re.IGNORECASE) for p in spark_patterns)
        
        return has_sql and has_spark
    
    def _is_config_file(self, file_path: Path, content: str) -> bool:
        """Check if this appears to be a configuration file."""
        config_indicators = [
            file_path.name.lower().endswith(('.conf', '.config', '.cfg')),
            re.search(r'^\s*[\w\-\.]+\s*[:=]', content, re.MULTILINE),
            re.search(r'^\s*\[[\w\s\-\.]+\]', content, re.MULTILINE),  # INI sections
        ]
        return any(config_indicators)
    
    def _detect_config_type(self, content: str) -> Tuple[FileType, float]:
        """Detect specific configuration file type."""
        # JSON-like structure
        if content.strip().startswith('{') and content.strip().endswith('}'):
            return (FileType.JSON, 0.7)
        
        # YAML-like structure
        if re.search(r'^\s*[\w\-]+\s*:\s*', content, re.MULTILINE):
            return (FileType.YAML, 0.7)
        
        # INI-like structure
        if re.search(r'^\s*\[[\w\s\-\.]+\]', content, re.MULTILINE):
            return (FileType.INI, 0.7)
        
        # Properties-like (key=value)
        if re.search(r'^\s*[\w\-\.]+\s*=', content, re.MULTILINE):
            return (FileType.PROPERTIES, 0.7)
        
        # Default to HOCON for other config-like content
        return (FileType.HOCON, 0.6)
    
    def _is_readable_text(self, content: str) -> bool:
        """Check if content is readable text."""
        if not content.strip():
            return False
        
        # Check if most characters are printable
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        printable_ratio = printable_chars / len(content) if content else 0
        
        return printable_ratio > 0.8