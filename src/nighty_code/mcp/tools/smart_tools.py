"""
Smart tools with context awareness and suggestions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import (
    ToolDefinition, ToolCategory, ToolParameter,
    MCPTool
)
from ..registry import register_tool
from .file_tools import BaseFileTool


@register_tool
class SmartSuggestTool(BaseFileTool, MCPTool):
    """Tool for getting context-aware suggestions."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="smart_suggest",
            description="Get context-aware suggestions for next actions",
            category=ToolCategory.NAVIGATION,
            parameters=[
                ToolParameter(
                    name="current_file",
                    type="string",
                    description="Current file being worked on",
                    required=False,
                    default=None
                ),
                ToolParameter(
                    name="recent_files",
                    type="array",
                    description="Recently accessed files",
                    required=False,
                    default=[]
                )
            ],
            returns="object",
            examples=[
                "smart_suggest(current_file='src/main.py')",
                "smart_suggest(recent_files=['README.md', 'setup.py'])"
            ]
        )
    
    async def execute(
        self,
        current_file: Optional[str] = None,
        recent_files: List[str] = None
    ) -> Dict[str, Any]:
        """Get smart suggestions based on context."""
        recent_files = recent_files or []
        
        # Get context suggestions
        suggestions = self.fuzzy_matcher.get_context_suggestions(current_file)
        
        # Add suggestions based on recent files
        if recent_files:
            # If working with tests, suggest more tests
            if any('test' in f.lower() for f in recent_files):
                suggestions['likely_queries'].append("Run all tests")
                suggestions['likely_queries'].append("Find failing tests")
            
            # If working with config files
            if any(f.endswith(('.json', '.yaml', '.yml', '.toml')) for f in recent_files):
                suggestions['likely_queries'].append("Validate configuration")
                suggestions['likely_queries'].append("Show environment variables")
        
        # Add general helpful suggestions
        suggestions['helpful_commands'] = [
            "list_directory - Explore project structure",
            "search_in_files - Find specific code or text",
            "find_files - Locate files by pattern",
            "read_file - Read file contents"
        ]
        
        # Get related files that actually exist
        if current_file:
            current_path = Path(current_file)
            
            # Find test file
            if not current_path.name.startswith('test_'):
                test_patterns = [
                    f"test_{current_path.stem}.py",
                    f"{current_path.stem}_test.py",
                    f"test_{current_path.name}"
                ]
                
                for pattern in test_patterns:
                    matches = self.fuzzy_matcher.find_similar_files(pattern, limit=1)
                    if matches:
                        suggestions['related_files'].insert(0, matches[0][0])
                        break
            
            # Find similar files
            similar = self.fuzzy_matcher.find_similar_files(
                current_path.stem, 
                limit=3,
                threshold=0.4
            )
            for file_path, score in similar:
                if file_path != current_file and file_path not in suggestions['related_files']:
                    suggestions['related_files'].append(file_path)
        
        # Limit suggestions
        suggestions['related_files'] = suggestions['related_files'][:5]
        suggestions['likely_queries'] = suggestions['likely_queries'][:5]
        
        return suggestions


@register_tool
class FuzzyFindTool(BaseFileTool, MCPTool):
    """Tool for finding files with fuzzy matching."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fuzzy_find",
            description="Find files using fuzzy name matching",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Fuzzy search query for file names"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of results",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="threshold",
                    type="number",
                    description="Similarity threshold (0.0-1.0)",
                    required=False,
                    default=0.6
                )
            ],
            returns="array",
            examples=[
                "fuzzy_find('readme')",
                "fuzzy_find('confg.yml')",  # Will find config.yml
                "fuzzy_find('main', limit=5)"
            ]
        )
    
    async def execute(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find files using fuzzy matching."""
        if not query or query.isspace():
            raise ValueError("Query cannot be empty")
        
        # Find similar files
        matches = self.fuzzy_matcher.find_similar_files(query, limit=limit, threshold=threshold)
        
        results = []
        for file_path, score in matches:
            # Get file info
            full_path = self.project_root / file_path
            
            file_info = {
                "file": file_path,
                "score": round(score, 3),
                "type": "directory" if full_path.is_dir() else "file"
            }
            
            if full_path.is_file():
                try:
                    file_info["size"] = full_path.stat().st_size
                except:
                    pass
            
            results.append(file_info)
        
        # If no matches, try to suggest corrections
        if not results:
            corrections = self.fuzzy_matcher.suggest_corrections(query)
            if corrections:
                return [{
                    "message": f"No exact matches for '{query}'",
                    "suggestions": corrections,
                    "hint": "Try searching for one of these suggestions"
                }]
        
        return results