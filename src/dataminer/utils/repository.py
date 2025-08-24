# src/dataminer/utils/repository.py
"""Repository analysis utilities for large-scale extraction"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import mimetypes
import hashlib
from datetime import datetime
import re
import json

from ..core.exceptions import RepositoryError
from ..core.types import FileAnalysis, RepositoryStructure


@dataclass 
class RepositoryAnalysis:
    """Results of repository analysis"""
    
    root_path: Path
    total_files: int = 0
    total_size_bytes: int = 0
    
    # File categorization
    code_files: List[Path] = field(default_factory=list)
    documentation_files: List[Path] = field(default_factory=list)
    config_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    data_files: List[Path] = field(default_factory=list)
    other_files: List[Path] = field(default_factory=list)
    
    # Analysis results
    languages: Dict[str, int] = field(default_factory=dict)
    frameworks: List[str] = field(default_factory=list)
    
    # Recommendations
    recommended_files: List[Path] = field(default_factory=list)
    extraction_priority: Dict[Path, float] = field(default_factory=dict)
    
    # Structure summary
    structure_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_duration_ms: float = 0.0


class RepositoryAnalyzer:
    """Analyzes repository structure for optimal extraction"""
    
    def __init__(self):
        self.language_extensions = {
            'python': ['.py', '.pyx', '.pyw'],
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.cxx', '.cc', '.hpp', '.hxx'],
            'csharp': ['.cs'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'swift': ['.swift'],
            'kotlin': ['.kt', '.kts'],
            'scala': ['.scala'],
            'r': ['.r', '.R'],
            'matlab': ['.m'],
            'shell': ['.sh', '.bash', '.zsh', '.fish'],
            'sql': ['.sql'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass', '.less'],
            'xml': ['.xml'],
            'yaml': ['.yml', '.yaml'],
            'json': ['.json'],
            'markdown': ['.md', '.markdown'],
            'tex': ['.tex'],
            'dockerfile': ['Dockerfile']
        }
        
        self.documentation_patterns = [
            r'readme.*',
            r'changelog.*',
            r'contributing.*',
            r'license.*',
            r'authors.*',
            r'credits.*',
            r'install.*',
            r'setup.*',
            r'.*\.md$',
            r'.*\.rst$',
            r'.*\.txt$',
            r'docs?/',
            r'documentation/'
        ]
        
        self.config_patterns = [
            r'.*\.json$',
            r'.*\.yml$',
            r'.*\.yaml$',
            r'.*\.toml$',
            r'.*\.ini$',
            r'.*\.cfg$',
            r'.*\.conf$',
            r'.*\.config$',
            r'package\.json$',
            r'requirements.*\.txt$',
            r'setup\.py$',
            r'setup\.cfg$',
            r'pyproject\.toml$',
            r'Makefile$',
            r'CMakeLists\.txt$',
            r'\.gitignore$',
            r'\.env.*'
        ]
        
        self.test_patterns = [
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'test.*\.js$',
            r'.*\.spec\.js$',
            r'.*\.test\.js$',
            r'tests?/',
            r'__tests__/',
            r'spec/',
            r'.*\.spec\.ts$',
            r'.*\.test\.ts$'
        ]
        
        # Common ignore patterns
        self.default_ignore_patterns = [
            r'\.git/',
            r'\.svn/',
            r'\.hg/',
            r'__pycache__/',
            r'\.pytest_cache/',
            r'node_modules/',
            r'\.venv/',
            r'venv/',
            r'\.env/',
            r'env/',
            r'build/',
            r'dist/',
            r'target/',
            r'\.DS_Store$',
            r'Thumbs\.db$',
            r'.*\.pyc$',
            r'.*\.pyo$',
            r'.*\.log$',
            r'.*\.tmp$',
            r'.*\.temp$'
        ]
    
    async def analyze_repository(
        self,
        repository_path: Path,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ) -> RepositoryAnalysis:
        """Analyze repository structure and recommend files for extraction"""
        
        start_time = datetime.now()
        
        if not repository_path.exists():
            raise RepositoryError(f"Repository path does not exist: {repository_path}")
        
        if not repository_path.is_dir():
            raise RepositoryError(f"Repository path is not a directory: {repository_path}")
        
        analysis = RepositoryAnalysis(root_path=repository_path)
        
        try:
            # Collect all files
            all_files = await self._collect_files(
                repository_path,
                file_patterns,
                exclude_patterns,
                max_file_size
            )
            
            analysis.total_files = len(all_files)
            
            # Categorize files
            await self._categorize_files(all_files, analysis)
            
            # Analyze languages and frameworks
            await self._analyze_languages(analysis)
            await self._detect_frameworks(analysis)
            
            # Calculate extraction priorities
            await self._calculate_extraction_priorities(analysis)
            
            # Select recommended files
            await self._select_recommended_files(analysis, max_files)
            
            # Generate structure summary
            await self._generate_structure_summary(analysis)
            
            # Calculate processing time
            end_time = datetime.now()
            analysis.analysis_duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return analysis
            
        except Exception as e:
            raise RepositoryError(f"Repository analysis failed: {str(e)}")
    
    async def _collect_files(
        self,
        root_path: Path,
        file_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
        max_file_size: int
    ) -> List[Path]:
        """Collect all relevant files in the repository"""
        
        collected_files = []
        ignore_patterns = self.default_ignore_patterns + (exclude_patterns or [])
        
        # Compile regex patterns for efficiency
        ignore_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in ignore_patterns]
        include_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in (file_patterns or [])]
        
        def should_ignore(file_path: Path) -> bool:
            relative_path = file_path.relative_to(root_path)
            path_str = str(relative_path)
            return any(regex.search(path_str) for regex in ignore_regexes)
        
        def should_include(file_path: Path) -> bool:
            if not include_regexes:  # No patterns means include all
                return True
            relative_path = file_path.relative_to(root_path)
            path_str = str(relative_path)
            return any(regex.search(path_str) for regex in include_regexes)
        
        try:
            for file_path in root_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Check ignore patterns
                if should_ignore(file_path):
                    continue
                
                # Check include patterns
                if not should_include(file_path):
                    continue
                
                # Check file size
                try:
                    if file_path.stat().st_size > max_file_size:
                        continue
                except OSError:
                    continue  # Skip files we can't stat
                
                # Check if file is readable
                if not self._is_text_file(file_path):
                    continue
                
                collected_files.append(file_path)
                
        except PermissionError as e:
            raise RepositoryError(f"Permission denied accessing repository: {e}")
        
        return collected_files
    
    async def _categorize_files(self, files: List[Path], analysis: RepositoryAnalysis):
        """Categorize files by type"""
        
        for file_path in files:
            file_name = file_path.name.lower()
            file_ext = file_path.suffix.lower()
            relative_path = file_path.relative_to(analysis.root_path)
            path_str = str(relative_path).lower()
            
            # Get file size
            try:
                file_size = file_path.stat().st_size
                analysis.total_size_bytes += file_size
            except OSError:
                file_size = 0
            
            # Categorize
            if any(re.search(pattern, path_str) for pattern in self.test_patterns):
                analysis.test_files.append(file_path)
            elif any(re.search(pattern, path_str) for pattern in self.documentation_patterns):
                analysis.documentation_files.append(file_path)
            elif any(re.search(pattern, path_str) for pattern in self.config_patterns):
                analysis.config_files.append(file_path)
            elif self._is_code_file(file_path):
                analysis.code_files.append(file_path)
            elif self._is_data_file(file_path):
                analysis.data_files.append(file_path)
            else:
                analysis.other_files.append(file_path)
    
    async def _analyze_languages(self, analysis: RepositoryAnalysis):
        """Analyze programming languages used"""
        
        language_counts = {}
        
        # Count by file extension
        for file_path in analysis.code_files:
            language = self._detect_language(file_path)
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1
        
        # Sort by frequency
        analysis.languages = dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True))
    
    async def _detect_frameworks(self, analysis: RepositoryAnalysis):
        """Detect frameworks and libraries used"""
        
        frameworks = set()
        
        # Check common framework indicators
        framework_indicators = {
            'django': ['manage.py', 'settings.py', 'wsgi.py'],
            'flask': ['app.py', 'application.py'],
            'fastapi': ['main.py'],
            'react': ['package.json', 'src/App.js', 'src/index.js'],
            'vue': ['package.json', 'src/App.vue'],
            'angular': ['angular.json', 'src/app/'],
            'spring': ['pom.xml', 'application.properties'],
            'rails': ['Gemfile', 'config/application.rb'],
            'laravel': ['artisan', 'composer.json'],
            'express': ['package.json', 'server.js', 'app.js'],
            'webpack': ['webpack.config.js'],
            'docker': ['Dockerfile', 'docker-compose.yml'],
            'kubernetes': ['*.yaml', '*.yml'],
            'terraform': ['*.tf'],
        }
        
        all_files = (analysis.code_files + analysis.config_files + 
                    analysis.documentation_files + analysis.other_files)
        
        for framework, indicators in framework_indicators.items():
            for indicator in indicators:
                if any(file_path.match(indicator) for file_path in all_files):
                    frameworks.add(framework)
                    break
        
        # Check package files for more specific detection
        package_files = [f for f in analysis.config_files 
                        if f.name in ['package.json', 'requirements.txt', 'Pipfile', 'poetry.lock']]
        
        for package_file in package_files:
            try:
                detected = await self._analyze_package_file(package_file)
                frameworks.update(detected)
            except Exception:
                pass  # Ignore errors in package file analysis
        
        analysis.frameworks = sorted(list(frameworks))
    
    async def _calculate_extraction_priorities(self, analysis: RepositoryAnalysis):
        """Calculate extraction priority scores for files"""
        
        all_files = (analysis.code_files + analysis.documentation_files + 
                    analysis.config_files + analysis.test_files + analysis.data_files)
        
        for file_path in all_files:
            priority = await self._calculate_file_priority(file_path, analysis)
            analysis.extraction_priority[file_path] = priority
    
    async def _calculate_file_priority(self, file_path: Path, analysis: RepositoryAnalysis) -> float:
        """Calculate extraction priority for a single file"""
        
        priority = 0.0
        file_name = file_path.name.lower()
        relative_path = file_path.relative_to(analysis.root_path)
        path_str = str(relative_path).lower()
        
        # Base priority by file type
        if file_path in analysis.documentation_files:
            priority += 0.8
            # Higher priority for README and main docs
            if 'readme' in file_name:
                priority += 0.2
            elif any(doc in file_name for doc in ['install', 'setup', 'getting-started']):
                priority += 0.15
        
        elif file_path in analysis.code_files:
            priority += 0.6
            # Higher priority for main files
            if file_name in ['main.py', 'app.py', 'index.js', 'main.js']:
                priority += 0.2
            # Higher priority for files in root or main directories
            elif len(relative_path.parts) <= 2:
                priority += 0.1
        
        elif file_path in analysis.config_files:
            priority += 0.4
            # Higher priority for key config files
            key_configs = ['package.json', 'requirements.txt', 'setup.py', 'makefile']
            if file_name in key_configs:
                priority += 0.3
        
        elif file_path in analysis.test_files:
            priority += 0.2
        
        # Size factor (prefer moderately sized files)
        try:
            file_size = file_path.stat().st_size
            if 1000 <= file_size <= 50000:  # 1KB to 50KB sweet spot
                priority += 0.1
            elif file_size > 100000:  # Large files get penalty
                priority -= 0.1
        except OSError:
            pass
        
        # Depth penalty (prefer files closer to root)
        depth = len(relative_path.parts)
        if depth <= 2:
            priority += 0.1
        elif depth > 4:
            priority -= 0.1
        
        # Language popularity bonus
        if analysis.languages:
            primary_language = list(analysis.languages.keys())[0]
            if self._detect_language(file_path) == primary_language:
                priority += 0.05
        
        return min(1.0, max(0.0, priority))
    
    async def _select_recommended_files(self, analysis: RepositoryAnalysis, max_files: Optional[int]):
        """Select recommended files for extraction"""
        
        # Sort by priority
        sorted_files = sorted(
            analysis.extraction_priority.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top files
        recommended_count = max_files or min(50, len(sorted_files))
        analysis.recommended_files = [file_path for file_path, _ in sorted_files[:recommended_count]]
    
    async def _generate_structure_summary(self, analysis: RepositoryAnalysis):
        """Generate repository structure summary"""
        
        analysis.structure_summary = {
            'total_files': analysis.total_files,
            'total_size_mb': analysis.total_size_bytes / (1024 * 1024),
            'file_types': {
                'code': len(analysis.code_files),
                'documentation': len(analysis.documentation_files),
                'configuration': len(analysis.config_files),
                'tests': len(analysis.test_files),
                'data': len(analysis.data_files),
                'other': len(analysis.other_files)
            },
            'languages': analysis.languages,
            'frameworks': analysis.frameworks,
            'recommended_files_count': len(analysis.recommended_files),
            'average_file_size_kb': (analysis.total_size_bytes / analysis.total_files / 1024) 
                                   if analysis.total_files > 0 else 0
        }
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file"""
        
        # Check by extension
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yml', '.yaml'}
        if file_path.suffix.lower() in text_extensions:
            return True
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type.startswith('text/'):
            return True
        
        # Check by reading first few bytes
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Simple heuristic: if most bytes are printable ASCII, likely text
                printable = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13])
                return printable / len(chunk) > 0.7 if chunk else True
        except (OSError, PermissionError):
            return False
    
    def _is_code_file(self, file_path: Path) -> bool:
        """Check if file is a source code file"""
        language = self._detect_language(file_path)
        return language is not None
    
    def _is_data_file(self, file_path: Path) -> bool:
        """Check if file is a data file"""
        data_extensions = {'.csv', '.json', '.xml', '.xlsx', '.tsv', '.dat', '.db', '.sqlite'}
        return file_path.suffix.lower() in data_extensions
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension"""
        
        file_ext = file_path.suffix.lower()
        file_name = file_path.name
        
        for language, extensions in self.language_extensions.items():
            if file_ext in extensions or file_name in extensions:
                return language
        
        return None
    
    async def _analyze_package_file(self, package_file: Path) -> Set[str]:
        """Analyze package file to detect frameworks"""
        
        frameworks = set()
        
        try:
            if package_file.name == 'package.json':
                with open(package_file, 'r') as f:
                    data = json.load(f)
                    
                dependencies = {}
                dependencies.update(data.get('dependencies', {}))
                dependencies.update(data.get('devDependencies', {}))
                
                framework_mappings = {
                    'react': ['react'],
                    'vue': ['vue'],
                    'angular': ['@angular/core'],
                    'express': ['express'],
                    'webpack': ['webpack'],
                    'babel': ['@babel/core'],
                    'typescript': ['typescript'],
                    'jest': ['jest'],
                    'eslint': ['eslint']
                }
                
                for framework, packages in framework_mappings.items():
                    if any(pkg in dependencies for pkg in packages):
                        frameworks.add(framework)
            
            elif package_file.name == 'requirements.txt':
                content = package_file.read_text()
                lines = [line.strip().split('==')[0].split('>=')[0].split('<')[0] 
                        for line in content.split('\n') if line.strip()]
                
                framework_mappings = {
                    'django': ['django'],
                    'flask': ['flask'],
                    'fastapi': ['fastapi'],
                    'pandas': ['pandas'],
                    'numpy': ['numpy'],
                    'tensorflow': ['tensorflow'],
                    'pytorch': ['torch']
                }
                
                for framework, packages in framework_mappings.items():
                    if any(pkg in lines for pkg in packages):
                        frameworks.add(framework)
        
        except Exception:
            pass  # Ignore errors in package analysis
        
        return frameworks
    
    def estimate_extraction_time(self, analysis: RepositoryAnalysis) -> float:
        """Estimate extraction time in seconds"""
        
        base_time_per_file = 2.0  # 2 seconds per file base time
        size_factor = analysis.total_size_bytes / (1024 * 1024)  # Size in MB
        
        # Adjust based on file types
        complexity_factor = 1.0
        if len(analysis.code_files) > len(analysis.documentation_files):
            complexity_factor = 1.5  # Code files are more complex to extract
        
        recommended_count = len(analysis.recommended_files)
        estimated_time = recommended_count * base_time_per_file * complexity_factor
        
        # Add size overhead
        estimated_time += size_factor * 0.5
        
        return max(10.0, estimated_time)  # Minimum 10 seconds