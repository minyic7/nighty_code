"""
Project Explorer - Mechanical exploration of project structure.

Phase 1 of the two-phase exploration system.
This module performs mechanical (non-LLM) exploration of the project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Result of project exploration."""
    file_tree: Dict
    file_contents: Dict[str, str]
    statistics: Dict
    patterns: Dict
    token_estimate: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "file_tree": self.file_tree,
            "file_contents": self.file_contents,
            "statistics": self.statistics,
            "patterns": self.patterns,
            "token_estimate": self.token_estimate
        }


class ProjectExplorer:
    """
    Mechanical project explorer - Phase 1.
    
    Explores project structure without using LLM.
    Collects raw data for LLM analysis.
    """
    
    # Files to skip
    SKIP_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'build', 'dist', '.pytest_cache', '.mypy_cache', 
        'htmlcov', '.tox', 'egg-info', '.idea', '.vscode'
    }
    
    SKIP_EXTENSIONS = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
        '.exe', '.bin', '.o', '.a', '.lib', '.png', '.jpg',
        '.jpeg', '.gif', '.ico', '.svg', '.pdf', '.zip',
        '.tar', '.gz', '.rar', '.7z', '.db', '.sqlite'
    }
    
    # Priority files to read (in order)
    PRIORITY_FILES = [
        # Documentation
        ['README.md', 'README.rst', 'README.txt', 'README'],
        # Configuration
        ['package.json', 'requirements.txt', 'pyproject.toml', 'setup.py',
         'go.mod', 'Cargo.toml', 'pom.xml', 'build.gradle'],
        # Environment
        ['.env.example', '.env.sample', 'config.yaml', 'config.json'],
        # Entry points
        ['main.py', 'app.py', 'index.js', 'index.ts', 'main.go',
         'server.py', 'run.py', '__main__.py', 'cli.py'],
        # CI/CD
        ['.github/workflows/main.yml', '.gitlab-ci.yml', 'Jenkinsfile',
         'Dockerfile', 'docker-compose.yml']
    ]
    
    def __init__(self, max_file_lines: int = 50, max_file_size: int = 10240):
        """
        Initialize explorer.
        
        Args:
            max_file_lines: Maximum lines to read per file
            max_file_size: Maximum file size in bytes
        """
        self.max_file_lines = max_file_lines
        self.max_file_size = max_file_size
        self.token_budget = 7200  # 90% of 8000
        self.tokens_used = 0
    
    def explore(self, project_path: Path) -> ExplorationResult:
        """
        Explore project structure.
        
        Args:
            project_path: Path to project root
            
        Returns:
            ExplorationResult with all exploration data
        """
        logger.info(f"Exploring project: {project_path}")
        
        # Reset token counter
        self.tokens_used = 0
        
        # Build file tree
        file_tree = self._build_tree(project_path)
        
        # Read priority files
        file_contents = self._read_priority_files(project_path)
        
        # Calculate statistics
        statistics = self._calculate_statistics(project_path, file_tree)
        
        # Detect patterns
        patterns = self._detect_patterns(project_path, file_tree)
        
        return ExplorationResult(
            file_tree=file_tree,
            file_contents=file_contents,
            statistics=statistics,
            patterns=patterns,
            token_estimate=self.tokens_used
        )
    
    def _build_tree(self, path: Path, depth: int = 0, max_depth: int = 5) -> Dict:
        """
        Build directory tree structure.
        
        Args:
            path: Current path
            depth: Current depth
            max_depth: Maximum depth to explore
            
        Returns:
            Tree structure as nested dict
        """
        if depth > max_depth:
            return {"...": "max depth reached"}
        
        tree = {}
        
        try:
            items = list(path.iterdir())
            
            # Sort: directories first, then files
            dirs = [d for d in items if d.is_dir() and d.name not in self.SKIP_DIRS]
            files = [f for f in items if f.is_file() and f.suffix not in self.SKIP_EXTENSIONS]
            
            # Limit items shown
            if len(items) > 100:
                tree["__info__"] = f"{len(items)} items (showing first 20)"
                dirs = dirs[:10]
                files = files[:10]
            elif len(items) > 20:
                tree["__info__"] = f"{len(items)} items (showing first 20)"
                total_shown = 20
                dirs = dirs[:min(10, total_shown)]
                files = files[:total_shown - len(dirs)]
            
            # Add directories
            for d in dirs:
                if self._estimate_tokens(tree) > self.token_budget * 0.3:
                    tree["..."] = "truncated for token limit"
                    break
                tree[d.name + "/"] = self._build_tree(d, depth + 1, max_depth)
            
            # Add files
            for f in files:
                if self._estimate_tokens(tree) > self.token_budget * 0.3:
                    tree["..."] = "truncated for token limit"
                    break
                size = f.stat().st_size
                tree[f.name] = f"{size:,} bytes"
        
        except PermissionError:
            tree["__error__"] = "Permission denied"
        except Exception as e:
            tree["__error__"] = str(e)
        
        return tree
    
    def _read_priority_files(self, project_path: Path) -> Dict[str, str]:
        """
        Read priority files in order.
        
        Args:
            project_path: Project root path
            
        Returns:
            Dictionary of file path -> content
        """
        contents = {}
        token_budget_for_files = self.token_budget * 0.6  # 60% for file contents
        
        for priority_group in self.PRIORITY_FILES:
            for filename in priority_group:
                if self.tokens_used > token_budget_for_files:
                    logger.info("Reached token budget for file reading")
                    break
                
                # Try to find file (case-insensitive on Windows)
                file_path = None
                for pattern in [filename, filename.lower(), filename.upper()]:
                    potential_path = project_path / pattern
                    if potential_path.exists() and potential_path.is_file():
                        file_path = potential_path
                        break
                
                if file_path:
                    content = self._read_file_safely(file_path)
                    if content:
                        relative_path = str(file_path.relative_to(project_path))
                        contents[relative_path] = content
                        self.tokens_used += self._estimate_tokens(content)
                        logger.debug(f"Read {relative_path}: {len(content)} chars")
        
        return contents
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """
        Safely read a file with limits.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content or None
        """
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                # Read only first part
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= self.max_file_lines:
                            lines.append(f"\n... (truncated at {self.max_file_lines} lines)")
                            break
                        lines.append(line)
                    return ''.join(lines)
            else:
                # Read entire file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    if len(lines) > self.max_file_lines:
                        lines = lines[:self.max_file_lines]
                        lines.append(f"... (truncated at {self.max_file_lines} lines)")
                        return '\n'.join(lines)
                    return content
                    
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return None
    
    def _calculate_statistics(self, project_path: Path, file_tree: Dict) -> Dict:
        """
        Calculate project statistics.
        
        Args:
            project_path: Project root
            file_tree: File tree structure
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_files": 0,
            "total_dirs": 0,
            "total_size_bytes": 0,
            "file_types": defaultdict(int),
            "languages": defaultdict(int)
        }
        
        # Language mapping
        ext_to_language = {
            '.py': 'Python', '.pyw': 'Python',
            '.js': 'JavaScript', '.jsx': 'JavaScript',
            '.ts': 'TypeScript', '.tsx': 'TypeScript',
            '.java': 'Java', '.class': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.c': 'C', '.h': 'C',
            '.cpp': 'C++', '.hpp': 'C++', '.cc': 'C++',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sql': 'SQL',
            '.sh': 'Shell', '.bash': 'Shell',
            '.yml': 'YAML', '.yaml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.html': 'HTML', '.htm': 'HTML',
            '.css': 'CSS', '.scss': 'CSS', '.sass': 'CSS',
            '.md': 'Markdown', '.rst': 'reStructuredText'
        }
        
        # Walk the project directory
        for root, dirs, files in os.walk(project_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            
            stats["total_dirs"] += len(dirs)
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip unwanted files
                if file_path.suffix in self.SKIP_EXTENSIONS:
                    continue
                
                stats["total_files"] += 1
                
                # File type statistics
                if file_path.suffix:
                    stats["file_types"][file_path.suffix] += 1
                    
                    # Language statistics
                    language = ext_to_language.get(file_path.suffix)
                    if language:
                        stats["languages"][language] += 1
                
                # Size statistics
                try:
                    stats["total_size_bytes"] += file_path.stat().st_size
                except:
                    pass
        
        # Convert defaultdicts to regular dicts
        stats["file_types"] = dict(stats["file_types"])
        stats["languages"] = dict(stats["languages"])
        
        # Sort by frequency
        stats["file_types"] = dict(sorted(
            stats["file_types"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10
        
        stats["languages"] = dict(sorted(
            stats["languages"].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Calculate percentages for languages
        total_lang_files = sum(stats["languages"].values())
        if total_lang_files > 0:
            stats["language_percentages"] = {
                lang: f"{(count/total_lang_files)*100:.1f}%"
                for lang, count in stats["languages"].items()
            }
        
        return stats
    
    def _detect_patterns(self, project_path: Path, file_tree: Dict) -> Dict:
        """
        Detect project patterns and structure.
        
        Args:
            project_path: Project root
            file_tree: File tree structure
            
        Returns:
            Detected patterns
        """
        patterns = {
            "project_type": "unknown",
            "framework": None,
            "build_tool": None,
            "test_framework": None,
            "ci_cd": None,
            "containerized": False,
            "version_control": False
        }
        
        # Check for various indicators
        root_files = set(f.name for f in project_path.iterdir() if f.is_file())
        root_dirs = set(d.name for d in project_path.iterdir() if d.is_dir())
        
        # Project type detection
        if 'package.json' in root_files:
            patterns["project_type"] = "Node.js"
            patterns["build_tool"] = "npm/yarn"
        elif 'requirements.txt' in root_files or 'setup.py' in root_files:
            patterns["project_type"] = "Python"
            patterns["build_tool"] = "pip"
        elif 'go.mod' in root_files:
            patterns["project_type"] = "Go"
            patterns["build_tool"] = "go modules"
        elif 'pom.xml' in root_files:
            patterns["project_type"] = "Java/Maven"
            patterns["build_tool"] = "Maven"
        elif 'build.gradle' in root_files or 'settings.gradle' in root_files:
            patterns["project_type"] = "Java/Gradle"
            patterns["build_tool"] = "Gradle"
        elif 'Cargo.toml' in root_files:
            patterns["project_type"] = "Rust"
            patterns["build_tool"] = "Cargo"
        
        # Framework detection
        if (project_path / 'manage.py').exists():
            patterns["framework"] = "Django"
        elif (project_path / 'app.py').exists() or (project_path / 'application.py').exists():
            if 'flask' in root_files or 'Flask' in str(file_tree):
                patterns["framework"] = "Flask"
        elif 'next.config.js' in root_files:
            patterns["framework"] = "Next.js"
        elif 'vue.config.js' in root_files:
            patterns["framework"] = "Vue.js"
        elif 'angular.json' in root_files:
            patterns["framework"] = "Angular"
        
        # Test framework
        if 'tests' in root_dirs or 'test' in root_dirs:
            patterns["test_framework"] = "detected"
        if 'pytest.ini' in root_files or 'conftest.py' in root_files:
            patterns["test_framework"] = "pytest"
        elif 'jest.config.js' in root_files:
            patterns["test_framework"] = "Jest"
        
        # CI/CD
        if '.github' in root_dirs:
            patterns["ci_cd"] = "GitHub Actions"
        elif '.gitlab-ci.yml' in root_files:
            patterns["ci_cd"] = "GitLab CI"
        elif 'Jenkinsfile' in root_files:
            patterns["ci_cd"] = "Jenkins"
        
        # Container
        if 'Dockerfile' in root_files or 'docker-compose.yml' in root_files:
            patterns["containerized"] = True
        
        # Version control
        if '.git' in root_dirs:
            patterns["version_control"] = True
        
        return patterns
    
    def _estimate_tokens(self, obj) -> int:
        """
        Estimate token count for an object.
        
        Rough estimate: 1 token ≈ 4 characters
        """
        if isinstance(obj, str):
            return len(obj) // 4
        elif isinstance(obj, dict):
            return len(json.dumps(obj)) // 4
        elif isinstance(obj, list):
            return sum(self._estimate_tokens(item) for item in obj)
        else:
            return len(str(obj)) // 4