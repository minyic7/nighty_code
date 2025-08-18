"""
Core data models and schemas for nighty_code.

This module defines the fundamental data structures used throughout the system
for file classification, identity cards, and metadata representation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from datetime import datetime


class FileType(Enum):
    """Enumeration of supported file types for classification."""
    
    # Programming Languages
    PYTHON = "python"
    JAVA = "java"
    SCALA = "scala"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    KOTLIN = "kotlin"
    
    # Query Languages
    SQL = "sql"
    HQL = "hql"  # Hive SQL
    GRAPHQL = "graphql"
    
    # Data Processing
    DBT = "dbt"
    SPARK_SQL = "spark_sql"
    
    # Configuration
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    XML = "xml"
    INI = "ini"
    PROPERTIES = "properties"
    HOCON = "hocon"
    
    # Infrastructure
    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    
    # CI/CD & Build
    GITHUB_WORKFLOW = "github_workflow"
    JENKINS = "jenkins"
    MAKEFILE = "makefile"
    SBT = "sbt"
    GRADLE = "gradle"
    MAVEN = "maven"
    NPM = "npm"
    
    # Documentation
    MARKDOWN = "markdown"
    RST = "rst"
    ASCIIDOC = "asciidoc"
    
    # Shell & Scripts
    BASH = "bash"
    POWERSHELL = "powershell"
    
    # Data Formats
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    
    # Unknown/Generic
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class Framework(Enum):
    """Enumeration of detected frameworks and libraries."""
    
    # Python Frameworks
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    PANDAS = "pandas"
    AIRFLOW = "airflow"
    
    # Java/Scala Frameworks
    SPRING = "spring"
    SPARK = "spark"
    AKKA = "akka"
    PLAY = "play"
    
    # JavaScript Frameworks
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NODE = "node"
    
    # Data Processing
    DBT_FRAMEWORK = "dbt"
    KAFKA = "kafka"
    
    # Testing Frameworks
    PYTEST = "pytest"
    JUNIT = "junit"
    JEST = "jest"
    
    UNKNOWN = "unknown"


class Complexity(Enum):
    """Code complexity levels."""
    TRIVIAL = "trivial"      # 0-0.2
    LOW = "low"              # 0.2-0.4
    MEDIUM = "medium"        # 0.4-0.6
    HIGH = "high"            # 0.6-0.8
    VERY_HIGH = "very_high"  # 0.8-1.0


@dataclass
class FileMetrics:
    """Basic file metrics and statistics."""
    
    size_bytes: int
    line_count: int
    non_empty_lines: int
    comment_lines: int = 0
    code_lines: int = 0
    
    # Content characteristics
    character_count: int = 0
    word_count: int = 0
    max_line_length: int = 0
    
    # Computed properties
    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @property
    def comment_ratio(self) -> float:
        """Ratio of comment lines to total lines."""
        return self.comment_lines / max(self.line_count, 1)
    
    @property
    def code_density(self) -> float:
        """Ratio of code lines to non-empty lines."""
        return self.code_lines / max(self.non_empty_lines, 1)


@dataclass
class ClassificationResult:
    """Result of file classification process."""
    
    file_path: Path
    file_type: FileType
    confidence: float  # 0.0 to 1.0
    
    # Detection methods used
    detected_by_extension: bool = False
    detected_by_content: bool = False
    detected_by_filename: bool = False
    detected_by_shebang: bool = False
    
    # Framework/library detection
    frameworks: List[Framework] = field(default_factory=list)
    
    # Content analysis
    language_version: Optional[str] = None
    encoding: str = "utf-8"
    
    # Metadata
    classification_timestamp: datetime = field(default_factory=datetime.now)
    classifier_version: str = "1.0.0"
    
    def add_framework(self, framework: Framework) -> None:
        """Add a detected framework."""
        if framework not in self.frameworks:
            self.frameworks.append(framework)


@dataclass
class ContentSignature:
    """Signature representing file content characteristics."""
    
    # Hash-based signatures
    content_hash: str
    structural_hash: str  # Hash of code structure (ignoring comments/whitespace)
    
    # Content patterns
    keywords: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    
    # Language-specific patterns
    language_patterns: Dict[str, List[str]] = field(default_factory=dict)
    
    @classmethod
    def from_content(cls, content: str, file_type: FileType) -> "ContentSignature":
        """Create content signature from file content."""
        return cls(
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            structural_hash=cls._compute_structural_hash(content, file_type)
        )
    
    @staticmethod
    def _compute_structural_hash(content: str, file_type: FileType) -> str:
        """Compute hash of code structure (implementation needed)."""
        # TODO: Implement structural hash computation
        # This should extract the logical structure and ignore formatting
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass 
class FileClassification:
    """Complete classification information for a file."""
    
    # Basic information
    file_path: Path
    classification_result: ClassificationResult
    metrics: FileMetrics
    content_signature: ContentSignature
    
    # Analysis metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analyzer_version: str = "1.0.0"
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    @property
    def is_valid(self) -> bool:
        """Check if classification is valid (no errors)."""
        return len(self.errors) == 0
    
    @property
    def complexity_level(self) -> Complexity:
        """Get complexity level based on metrics."""
        # TODO: Implement complexity calculation based on metrics
        # This is a placeholder
        if self.metrics.line_count < 50:
            return Complexity.TRIVIAL
        elif self.metrics.line_count < 200:
            return Complexity.LOW
        elif self.metrics.line_count < 500:
            return Complexity.MEDIUM
        elif self.metrics.line_count < 1000:
            return Complexity.HIGH
        else:
            return Complexity.VERY_HIGH