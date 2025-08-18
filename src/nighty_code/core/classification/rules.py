"""
Classification rules and mappings for file type detection.
"""

from typing import Dict, List, Set, Pattern
import re
from ..models import FileType, Framework


class ClassificationRules:
    """Central repository for file classification rules and patterns."""
    
    def __init__(self):
        self._extension_mapping = self._build_extension_mapping()
        self._filename_patterns = self._build_filename_patterns()
        self._shebang_patterns = self._build_shebang_patterns()
        self._content_patterns = self._build_content_patterns()
        self._framework_patterns = self._build_framework_patterns()
    
    @property
    def extension_mapping(self) -> Dict[str, FileType]:
        """Get file extension to file type mapping."""
        return self._extension_mapping.copy()
    
    @property
    def filename_patterns(self) -> Dict[Pattern, FileType]:
        """Get filename regex patterns to file type mapping."""
        return self._filename_patterns.copy()
    
    @property
    def shebang_patterns(self) -> Dict[Pattern, FileType]:
        """Get shebang patterns to file type mapping."""
        return self._shebang_patterns.copy()
    
    @property
    def content_patterns(self) -> Dict[FileType, List[Pattern]]:
        """Get content patterns for each file type."""
        return {k: v.copy() for k, v in self._content_patterns.items()}
    
    @property
    def framework_patterns(self) -> Dict[Framework, List[Pattern]]:
        """Get framework detection patterns."""
        return {k: v.copy() for k, v in self._framework_patterns.items()}
    
    def _build_extension_mapping(self) -> Dict[str, FileType]:
        """Build mapping from file extensions to file types."""
        return {
            # Programming Languages
            '.py': FileType.PYTHON,
            '.pyw': FileType.PYTHON,
            '.java': FileType.JAVA,
            '.scala': FileType.SCALA,
            '.sc': FileType.SCALA,
            '.js': FileType.JAVASCRIPT,
            '.mjs': FileType.JAVASCRIPT,
            '.jsx': FileType.JAVASCRIPT,
            '.ts': FileType.TYPESCRIPT,
            '.tsx': FileType.TYPESCRIPT,
            '.go': FileType.GO,
            '.rs': FileType.RUST,
            '.cpp': FileType.CPP,
            '.cxx': FileType.CPP,
            '.cc': FileType.CPP,
            '.hpp': FileType.CPP,
            '.c': FileType.C,
            '.h': FileType.C,
            '.cs': FileType.CSHARP,
            '.kt': FileType.KOTLIN,
            '.kts': FileType.KOTLIN,
            
            # Query Languages
            '.sql': FileType.SQL,
            '.hql': FileType.HQL,
            '.hive': FileType.HQL,
            '.graphql': FileType.GRAPHQL,
            '.gql': FileType.GRAPHQL,
            
            # Configuration
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.json': FileType.JSON,
            '.toml': FileType.TOML,
            '.xml': FileType.XML,
            '.ini': FileType.INI,
            '.properties': FileType.PROPERTIES,
            '.conf': FileType.HOCON,
            '.config': FileType.HOCON,
            
            # Build & CI/CD
            '.sbt': FileType.SBT,
            '.gradle': FileType.GRADLE,
            '.pom': FileType.MAVEN,
            
            # Documentation
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.rst': FileType.RST,
            '.adoc': FileType.ASCIIDOC,
            
            # Shell & Scripts
            '.sh': FileType.BASH,
            '.bash': FileType.BASH,
            '.zsh': FileType.BASH,
            '.ps1': FileType.POWERSHELL,
            
            # Data Formats
            '.csv': FileType.CSV,
            '.parquet': FileType.PARQUET,
            '.avro': FileType.AVRO,
            
            # Generic
            '.txt': FileType.TEXT,
        }
    
    def _build_filename_patterns(self) -> Dict[Pattern, FileType]:
        """Build filename-based detection patterns."""
        patterns = {}
        
        # Add compiled regex patterns
        filename_rules = [
            # Infrastructure
            (r'^Dockerfile$', FileType.DOCKERFILE),
            (r'^Dockerfile\..+$', FileType.DOCKERFILE),
            (r'^docker-compose\.ya?ml$', FileType.DOCKER_COMPOSE),
            (r'\.terraform\..*', FileType.TERRAFORM),
            (r'^.*\.tf$', FileType.TERRAFORM),
            
            # Build files
            (r'^Makefile$', FileType.MAKEFILE),
            (r'^makefile$', FileType.MAKEFILE),
            (r'^build\.gradle$', FileType.GRADLE),
            (r'^pom\.xml$', FileType.MAVEN),
            (r'^package\.json$', FileType.NPM),
            
            # GitHub workflows
            (r'^\.github/workflows/.*\.ya?ml$', FileType.GITHUB_WORKFLOW),
            
            # DBT
            (r'^dbt_project\.yml$', FileType.DBT),
            (r'.*/models/.*\.sql$', FileType.DBT),
            (r'.*/macros/.*\.sql$', FileType.DBT),
            
            # Kubernetes
            (r'.*k8s.*\.ya?ml$', FileType.KUBERNETES),
            (r'.*kubernetes.*\.ya?ml$', FileType.KUBERNETES),
        ]
        
        for pattern_str, file_type in filename_rules:
            patterns[re.compile(pattern_str, re.IGNORECASE)] = file_type
        
        return patterns
    
    def _build_shebang_patterns(self) -> Dict[Pattern, FileType]:
        """Build shebang-based detection patterns."""
        patterns = {}
        
        shebang_rules = [
            (r'#!/bin/bash', FileType.BASH),
            (r'#!/bin/sh', FileType.BASH),
            (r'#!/usr/bin/env bash', FileType.BASH),
            (r'#!/usr/bin/env sh', FileType.BASH),
            (r'#!/usr/bin/env python', FileType.PYTHON),
            (r'#!/usr/bin/python', FileType.PYTHON),
            (r'#!/usr/bin/env node', FileType.JAVASCRIPT),
            (r'#!/usr/bin/node', FileType.JAVASCRIPT),
        ]
        
        for pattern_str, file_type in shebang_rules:
            patterns[re.compile(pattern_str)] = file_type
        
        return patterns
    
    def _build_content_patterns(self) -> Dict[FileType, List[Pattern]]:
        """Build content-based detection patterns."""
        patterns = {}
        
        # Python patterns
        patterns[FileType.PYTHON] = [
            re.compile(r'\bimport\s+\w+'),
            re.compile(r'\bfrom\s+\w+\s+import\s+'),
            re.compile(r'\bdef\s+\w+\s*\('),
            re.compile(r'\bclass\s+\w+\s*(\(.*\))?\s*:'),
            re.compile(r'\bif\s+__name__\s*==\s*["\']__main__["\']'),
        ]
        
        # Java patterns
        patterns[FileType.JAVA] = [
            re.compile(r'\bpublic\s+class\s+\w+'),
            re.compile(r'\bimport\s+[\w.]+;'),
            re.compile(r'\bpublic\s+static\s+void\s+main\s*\('),
            re.compile(r'\bpackage\s+[\w.]+;'),
        ]
        
        # Scala patterns
        patterns[FileType.SCALA] = [
            re.compile(r'\bobject\s+\w+'),
            re.compile(r'\bclass\s+\w+'),
            re.compile(r'\btrait\s+\w+'),
            re.compile(r'\bdef\s+\w+'),
            re.compile(r'\bval\s+\w+'),
            re.compile(r'\bvar\s+\w+'),
            re.compile(r'\bimport\s+[\w.]+'),
        ]
        
        # SQL patterns
        patterns[FileType.SQL] = [
            re.compile(r'\bSELECT\s+', re.IGNORECASE),
            re.compile(r'\bFROM\s+', re.IGNORECASE),
            re.compile(r'\bWHERE\s+', re.IGNORECASE),
            re.compile(r'\bINSERT\s+INTO\s+', re.IGNORECASE),
            re.compile(r'\bUPDATE\s+', re.IGNORECASE),
            re.compile(r'\bDELETE\s+FROM\s+', re.IGNORECASE),
            re.compile(r'\bCREATE\s+TABLE\s+', re.IGNORECASE),
        ]
        
        # Hive SQL patterns
        patterns[FileType.HQL] = [
            re.compile(r'\bPARTITIONED\s+BY\s+', re.IGNORECASE),
            re.compile(r'\bSTORED\s+AS\s+', re.IGNORECASE),
            re.compile(r'\bLOCATION\s+', re.IGNORECASE),
            re.compile(r'\bMSCK\s+REPAIR\s+TABLE\s+', re.IGNORECASE),
            re.compile(r'\bSHOW\s+PARTITIONS\s+', re.IGNORECASE),
        ]
        
        # JavaScript patterns
        patterns[FileType.JAVASCRIPT] = [
            re.compile(r'\bfunction\s+\w+\s*\('),
            re.compile(r'\bconst\s+\w+\s*='),
            re.compile(r'\blet\s+\w+\s*='),
            re.compile(r'\brequire\s*\('),
            re.compile(r'\bmodule\.exports\s*='),
            re.compile(r'\bexport\s+'),
        ]
        
        # YAML patterns
        patterns[FileType.YAML] = [
            re.compile(r'^[\w-]+\s*:\s*'),
            re.compile(r'^\s*-\s+\w+'),
            re.compile(r'---\s*$'),
        ]
        
        # JSON patterns  
        patterns[FileType.JSON] = [
            re.compile(r'^\s*{'),
            re.compile(r'}\s*$'),
            re.compile(r'"\w+"\s*:\s*'),
        ]
        
        return patterns
    
    def _build_framework_patterns(self) -> Dict[Framework, List[Pattern]]:
        """Build framework detection patterns."""
        patterns = {}
        
        # Python frameworks
        patterns[Framework.DJANGO] = [
            re.compile(r'\bfrom\s+django\b'),
            re.compile(r'\bimport\s+django\b'),
            re.compile(r'\bDjango\b'),
        ]
        
        patterns[Framework.FLASK] = [
            re.compile(r'\bfrom\s+flask\b'),
            re.compile(r'\bimport\s+flask\b'),
            re.compile(r'\bFlask\b'),
        ]
        
        patterns[Framework.FASTAPI] = [
            re.compile(r'\bfrom\s+fastapi\b'),
            re.compile(r'\bimport\s+fastapi\b'),
            re.compile(r'\bFastAPI\b'),
        ]
        
        patterns[Framework.PANDAS] = [
            re.compile(r'\bimport\s+pandas\b'),
            re.compile(r'\bpd\.'),
            re.compile(r'\bDataFrame\b'),
        ]
        
        patterns[Framework.AIRFLOW] = [
            re.compile(r'\bfrom\s+airflow\b'),
            re.compile(r'\bimport\s+airflow\b'),
            re.compile(r'\bDAG\b'),
        ]
        
        # Java/Scala frameworks
        patterns[Framework.SPARK] = [
            re.compile(r'\bspark\b', re.IGNORECASE),
            re.compile(r'\bSparkSession\b'),
            re.compile(r'\bDataFrame\b'),
            re.compile(r'\borg\.apache\.spark\b'),
        ]
        
        patterns[Framework.SPRING] = [
            re.compile(r'\bspring\b', re.IGNORECASE),
            re.compile(r'\b@Component\b'),
            re.compile(r'\b@Service\b'),
            re.compile(r'\b@RestController\b'),
        ]
        
        patterns[Framework.AKKA] = [
            re.compile(r'\bakka\b', re.IGNORECASE),
            re.compile(r'\bActorSystem\b'),
            re.compile(r'\bActor\b'),
        ]
        
        # Data processing
        patterns[Framework.DBT_FRAMEWORK] = [
            re.compile(r'\{\{\s*config\s*\('),
            re.compile(r'\{\{\s*ref\s*\('),
            re.compile(r'\{\{\s*source\s*\('),
        ]
        
        patterns[Framework.KAFKA] = [
            re.compile(r'\bkafka\b', re.IGNORECASE),
            re.compile(r'\bKafkaConsumer\b'),
            re.compile(r'\bKafkaProducer\b'),
        ]
        
        return patterns
    
    def get_file_type_keywords(self, file_type: FileType) -> Set[str]:
        """Get common keywords for a file type."""
        keyword_mapping = {
            FileType.PYTHON: {'def', 'class', 'import', 'from', 'if', 'for', 'while', 'try', 'except'},
            FileType.JAVA: {'public', 'private', 'class', 'interface', 'import', 'package', 'static'},
            FileType.SCALA: {'object', 'class', 'trait', 'def', 'val', 'var', 'import', 'package'},
            FileType.JAVASCRIPT: {'function', 'const', 'let', 'var', 'import', 'export', 'require'},
            FileType.SQL: {'select', 'from', 'where', 'insert', 'update', 'delete', 'create', 'table'},
            FileType.HQL: {'select', 'from', 'where', 'partitioned', 'stored', 'location'},
        }
        return keyword_mapping.get(file_type, set())
    
    def get_comment_patterns(self, file_type: FileType) -> List[Pattern]:
        """Get comment patterns for a file type."""
        comment_mapping = {
            FileType.PYTHON: [re.compile(r'#.*$', re.MULTILINE)],
            FileType.JAVA: [
                re.compile(r'//.*$', re.MULTILINE),
                re.compile(r'/\*.*?\*/', re.DOTALL)
            ],
            FileType.SCALA: [
                re.compile(r'//.*$', re.MULTILINE),
                re.compile(r'/\*.*?\*/', re.DOTALL)
            ],
            FileType.JAVASCRIPT: [
                re.compile(r'//.*$', re.MULTILINE),
                re.compile(r'/\*.*?\*/', re.DOTALL)
            ],
            FileType.SQL: [re.compile(r'--.*$', re.MULTILINE)],
            FileType.HQL: [re.compile(r'--.*$', re.MULTILINE)],
            FileType.BASH: [re.compile(r'#.*$', re.MULTILINE)],
            FileType.YAML: [re.compile(r'#.*$', re.MULTILINE)],
        }
        return comment_mapping.get(file_type, [])