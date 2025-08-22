"""
Fuzzy matching and smart suggestions for MCP tools.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher, get_close_matches
import re


class FuzzyMatcher:
    """Provides fuzzy matching and suggestions for file names."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._file_cache: Optional[List[str]] = None
        self._importance_patterns = {
            'readme': 10,
            'main': 8,
            'index': 8,
            'config': 7,
            'setup': 7,
            'package': 6,
            'requirements': 6,
            '__init__': 5,
            'test': 3,
            'example': 2,
            'demo': 2,
        }
    
    def find_similar_files(self, query: str, limit: int = 5, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find files with names similar to the query.
        
        Returns list of (file_path, similarity_score) tuples.
        """
        if self._file_cache is None:
            self._build_file_cache()
        
        # Normalize query
        query_lower = query.lower().strip()
        query_parts = self._split_path(query)
        
        results = []
        
        for file_path in self._file_cache:
            # Calculate similarity
            score = self._calculate_similarity(query_lower, file_path.lower())
            
            # Boost score for exact filename matches
            file_name = Path(file_path).name
            if query_lower in file_name.lower():
                score = min(1.0, score + 0.3)
            
            # Check path components
            path_parts = self._split_path(file_path)
            if self._path_components_match(query_parts, path_parts):
                score = min(1.0, score + 0.2)
            
            if score >= threshold:
                results.append((file_path, score))
        
        # Sort by score (descending) and path length (ascending for ties)
        results.sort(key=lambda x: (-x[1], len(x[0])))
        
        return results[:limit]
    
    def suggest_corrections(self, invalid_path: str) -> List[str]:
        """
        Suggest corrections for an invalid file path.
        
        Returns list of suggested paths.
        """
        # Try different correction strategies
        suggestions = []
        
        # 1. Direct fuzzy match
        similar = self.find_similar_files(invalid_path, limit=3, threshold=0.5)
        suggestions.extend([path for path, _ in similar])
        
        # 2. Try fixing common typos
        typo_fixed = self._fix_common_typos(invalid_path)
        if typo_fixed != invalid_path:
            similar = self.find_similar_files(typo_fixed, limit=2, threshold=0.6)
            suggestions.extend([path for path, _ in similar if path not in suggestions])
        
        # 3. Try different extensions
        base_name = Path(invalid_path).stem
        for ext in ['.py', '.txt', '.md', '.json', '.yaml', '.yml', '.js', '.ts']:
            test_path = str(Path(invalid_path).parent / (base_name + ext))
            if self._file_exists_in_cache(test_path) and test_path not in suggestions:
                suggestions.append(test_path)
        
        return suggestions[:5]
    
    def rank_search_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rank search results by relevance.
        
        Considers:
        - File importance (README, main, etc.)
        - Path depth (prefer shallower)
        - File type relevance
        - Query match quality
        """
        ranked_results = []
        
        for result in results:
            file_path = result.get('file', '')
            score = 0.0
            
            # File importance
            file_name = Path(file_path).name.lower()
            for pattern, importance in self._importance_patterns.items():
                if pattern in file_name:
                    score += importance * 0.1
                    break
            
            # Path depth (prefer shallower)
            depth = len(Path(file_path).parts)
            score -= depth * 0.05
            
            # File type relevance
            ext = Path(file_path).suffix
            if ext in ['.py', '.js', '.ts', '.java', '.go']:
                score += 0.3  # Code files
            elif ext in ['.md', '.txt', '.rst']:
                score += 0.2  # Documentation
            elif ext in ['.json', '.yaml', '.yml', '.toml']:
                score += 0.15  # Config files
            
            # Query match quality in content
            content = result.get('content', '')
            if content:
                # Exact match
                if query in content:
                    score += 0.5
                # Case insensitive match
                elif query.lower() in content.lower():
                    score += 0.3
                # Word boundary match
                if re.search(r'\b' + re.escape(query) + r'\b', content, re.IGNORECASE):
                    score += 0.2
            
            # Line number (prefer earlier matches)
            line_num = result.get('line', 99999)
            if line_num < 100:
                score += 0.1
            
            ranked_results.append({**result, '_score': score})
        
        # Sort by score
        ranked_results.sort(key=lambda x: x['_score'], reverse=True)
        
        # Remove score from results
        for result in ranked_results:
            result.pop('_score', None)
        
        return ranked_results
    
    def get_context_suggestions(self, current_file: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get context-aware suggestions based on current file.
        
        Returns suggestions for:
        - Related files (tests, configs, etc.)
        - Common next actions
        - Relevant commands
        """
        suggestions = {
            'related_files': [],
            'likely_queries': [],
            'relevant_commands': []
        }
        
        if not current_file:
            # General suggestions
            suggestions['related_files'] = [
                'README.md',
                'setup.py',
                'package.json',
                'requirements.txt',
                '.env'
            ]
            suggestions['likely_queries'] = [
                'Show project structure',
                'Find main entry point',
                'List configuration files'
            ]
            return suggestions
        
        current_path = Path(current_file)
        
        # Find related files
        if current_path.suffix == '.py':
            # Python file - suggest test file
            test_name = f"test_{current_path.name}"
            suggestions['related_files'].append(f"tests/{test_name}")
            suggestions['related_files'].append(f"test/{test_name}")
            
            # Suggest __init__.py in same directory
            init_file = current_path.parent / "__init__.py"
            suggestions['related_files'].append(str(init_file))
            
            suggestions['likely_queries'] = [
                f"Find usages of functions in {current_path.name}",
                f"Show imports in {current_path.name}",
                f"Find test for {current_path.name}"
            ]
        
        elif current_path.name in ['package.json', 'requirements.txt', 'setup.py']:
            # Dependency file - suggest related configs
            suggestions['related_files'] = [
                '.env',
                '.env.example',
                'README.md',
                'Dockerfile',
                '.gitignore'
            ]
            suggestions['likely_queries'] = [
                "Show installed dependencies",
                "Find security vulnerabilities",
                "Check for updates"
            ]
        
        elif current_path.suffix in ['.md', '.rst', '.txt']:
            # Documentation - suggest related docs
            suggestions['related_files'] = [
                'README.md',
                'CONTRIBUTING.md',
                'LICENSE',
                'CHANGELOG.md'
            ]
            suggestions['likely_queries'] = [
                "Find code examples",
                "Show related documentation",
                "Find references to this file"
            ]
        
        # Filter suggestions to existing files
        if self._file_cache:
            suggestions['related_files'] = [
                f for f in suggestions['related_files']
                if self._file_exists_in_cache(f)
            ][:5]
        
        return suggestions
    
    def _build_file_cache(self):
        """Build cache of all files in project."""
        self._file_cache = []
        
        # Limit cache building for performance
        MAX_FILES = 10000
        EXCLUDED_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv'}
        
        count = 0
        for path in self.project_root.rglob('*'):
            if count >= MAX_FILES:
                break
            
            # Skip excluded directories
            if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
                continue
            
            if path.is_file():
                try:
                    rel_path = path.relative_to(self.project_root)
                    self._file_cache.append(str(rel_path))
                    count += 1
                except:
                    continue
        
        # Normalize paths to use forward slashes
        self._file_cache = [path.replace('\\', '/') for path in self._file_cache]
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _split_path(self, path: str) -> List[str]:
        """Split path into components."""
        parts = []
        p = Path(path)
        
        # Get all parts
        parts.extend(p.parts)
        
        # Also split on common separators in filename
        if p.name:
            # Split on _, -, .
            name_parts = re.split(r'[_.\-]', p.stem)
            parts.extend(name_parts)
        
        return [p.lower() for p in parts if p]
    
    def _path_components_match(self, query_parts: List[str], path_parts: List[str]) -> bool:
        """Check if query components match path components."""
        for part in query_parts:
            if part in path_parts:
                return True
        return False
    
    def _fix_common_typos(self, path: str) -> str:
        """Fix common typos in file paths."""
        corrections = {
            'teh': 'the',
            'conifg': 'config',
            'cofnig': 'config',
            'pacakge': 'package',
            'packge': 'package',
            'requirments': 'requirements',
            'reqiurements': 'requirements',
            'pyhton': 'python',
            'pytohn': 'python',
            'mian': 'main',
            'tset': 'test',
            'scr': 'src',
            'lgo': 'log',
            'josn': 'json',
            'ymal': 'yaml',
        }
        
        result = path.lower()
        for typo, correct in corrections.items():
            result = result.replace(typo, correct)
        
        return result
    
    def _file_exists_in_cache(self, path: str) -> bool:
        """Check if file exists in cache."""
        if not self._file_cache:
            return False
        
        # Normalize path - convert to forward slashes for comparison
        norm_path = str(Path(path)).replace('\\', '/')
        
        return any(
            norm_path == cached_path.replace('\\', '/') or 
            Path(norm_path).name == Path(cached_path).name
            for cached_path in self._file_cache
        )