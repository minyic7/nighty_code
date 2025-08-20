"""
LLM-based code parser for languages without dedicated parsers.

This module provides code parsing capabilities using LLMs for languages
that don't have tree-sitter support or when more semantic understanding is needed.
"""

from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class LLMParser:
    """Parse code using Large Language Models for entity and relationship extraction."""
    
    def __init__(self, llm_client):
        """
        Initialize the LLM parser.
        
        Args:
            llm_client: An LLM client (e.g., AnthropicClient or OpenAIClient)
        """
        self.llm_client = llm_client
        self.model = "claude-3-5-haiku-20241022"  # Default model
    
    def parse_code(
        self, 
        code: str, 
        language: str, 
        file_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse code to extract entities and relationships.
        
        Args:
            code: Source code to parse
            language: Programming language
            file_path: Optional file path for context
            
        Returns:
            Dictionary containing entities and relationships, or None if parsing fails
        """
        
        if not code or not code.strip():
            return {"entities": [], "relationships": []}
        
        # Truncate very long files to avoid token limits
        max_chars = 50000
        if len(code) > max_chars:
            code = code[:max_chars] + "\n... (truncated)"
        
        try:
            # Create parsing prompt
            prompt = self._create_parsing_prompt(code, language, file_path)
            
            # Call LLM
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = self.llm_client.create_message(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract JSON from response
            if response and "content" in response:
                content = ""
                for block in response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
                
                # Try to extract JSON from the content
                result = self._extract_json(content)
                
                if result:
                    logger.debug(f"Parsed {len(result.get('entities', []))} entities from {file_path}")
                    return result
            
            return {"entities": [], "relationships": []}
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path} with LLM: {e}")
            return {"entities": [], "relationships": []}
    
    def _create_parsing_prompt(self, code: str, language: str, file_path: Optional[str]) -> str:
        """Create a prompt for code parsing."""
        
        file_context = f" (file: {file_path})" if file_path else ""
        
        return f"""Parse the following {language} code{file_context} and extract entities and relationships.

Return ONLY a JSON object with this structure:
{{
  "entities": [
    {{
      "name": "EntityName",
      "type": "class|function|method|interface|enum|variable|constant",
      "line_start": 10,
      "line_end": 20,
      "qualified_name": "module.ClassName.methodName",
      "description": "Brief description"
    }}
  ],
  "relationships": [
    {{
      "source": "SourceEntity",
      "target": "TargetEntity", 
      "type": "inherits|implements|calls|uses|imports",
      "line": 15
    }}
  ]
}}

Code to parse:
```{language}
{code}
```

Important:
- Extract all major entities (classes, functions, methods, interfaces, enums)
- Include line numbers when possible
- For relationships, identify inheritance, implementations, method calls, and imports
- Return ONLY valid JSON, no explanation text"""
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response text."""
        
        # Try to find JSON in the text
        import re
        
        # Look for JSON block
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                # Try to parse as JSON
                result = json.loads(match)
                
                # Validate structure
                if isinstance(result, dict):
                    if "entities" not in result:
                        result["entities"] = []
                    if "relationships" not in result:
                        result["relationships"] = []
                    
                    # Ensure lists
                    if not isinstance(result["entities"], list):
                        result["entities"] = []
                    if not isinstance(result["relationships"], list):
                        result["relationships"] = []
                    
                    return result
                    
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to parse the entire text
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except:
            pass
        
        logger.warning("Could not extract valid JSON from LLM response")
        return None
    
    def parse_directory(
        self, 
        directory_path: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Parse all files in a directory.
        
        Args:
            directory_path: Path to directory
            extensions: File extensions to parse (e.g., ['.py', '.js'])
            
        Returns:
            Dictionary with file paths as keys and parse results as values
        """
        from pathlib import Path
        
        results = {}
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        # Default extensions for common languages
        if extensions is None:
            extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', 
                '.cpp', '.c', '.h', '.cs', '.rb', '.go', '.rs',
                '.kt', '.swift', '.php', '.r', '.m'
            ]
        
        # Find all matching files
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Determine language from extension
                    language = self._get_language_from_extension(ext)
                    
                    # Parse the file
                    result = self.parse_code(code, language, str(file_path))
                    
                    if result:
                        results[str(file_path)] = result
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
        
        return results
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Map file extension to language name."""
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.php': 'php',
            '.r': 'r',
            '.m': 'matlab'
        }
        
        return language_map.get(ext.lower(), 'unknown')