"""
Project Analyzer - LLM-powered analysis and compression.

Phase 2 of the two-phase exploration system.
This module uses LLM to understand and compress project data.
"""

import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ProjectAnalysis:
    """Analysis result from LLM processing."""
    project_type: str
    purpose: str
    tech_stack: Dict[str, list]
    key_features: list
    architecture: str
    entry_points: list
    setup_instructions: str
    compressed_overview: str  # The final 2000-token overview
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "project_type": self.project_type,
            "purpose": self.purpose,
            "tech_stack": self.tech_stack,
            "key_features": self.key_features,
            "architecture": self.architecture,
            "entry_points": self.entry_points,
            "setup_instructions": self.setup_instructions,
            "compressed_overview": self.compressed_overview
        }


class ProjectAnalyzer:
    """
    LLM-powered project analyzer - Phase 2.
    
    Takes raw exploration data and creates intelligent summaries.
    Target: Compress ~7000 tokens to ~2000 tokens with high value.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize analyzer with LLM client.
        
        Args:
            llm_client: LLM client for analysis
        """
        self.llm_client = llm_client
        self.max_overview_tokens = 2000  # Target for final overview
    
    def analyze(self, exploration_data: Dict) -> ProjectAnalysis:
        """
        Analyze and compress exploration data.
        
        Args:
            exploration_data: Raw data from ProjectExplorer
            
        Returns:
            ProjectAnalysis with compressed understanding
        """
        logger.info("Starting LLM analysis of project data")
        
        # Extract components from exploration data
        file_tree = exploration_data.get("file_tree", {})
        file_contents = exploration_data.get("file_contents", {})
        statistics = exploration_data.get("statistics", {})
        patterns = exploration_data.get("patterns", {})
        
        # Progressive analysis steps
        
        # 1. Understand project type and purpose
        project_understanding = self._analyze_project_type(
            file_tree, file_contents, statistics, patterns
        )
        
        # 2. Extract tech stack
        tech_stack = self._analyze_tech_stack(
            file_contents, statistics, patterns
        )
        
        # 3. Identify key features
        key_features = self._extract_key_features(
            file_contents, file_tree
        )
        
        # 4. Understand architecture
        architecture = self._analyze_architecture(
            file_tree, patterns, statistics
        )
        
        # 5. Find entry points
        entry_points = self._identify_entry_points(
            file_contents, patterns
        )
        
        # 6. Extract setup instructions
        setup_instructions = self._extract_setup_instructions(
            file_contents
        )
        
        # 7. Create final compressed overview
        compressed_overview = self._create_compressed_overview(
            project_understanding,
            tech_stack,
            key_features,
            architecture,
            entry_points,
            setup_instructions,
            statistics
        )
        
        return ProjectAnalysis(
            project_type=project_understanding.get("type", "Unknown"),
            purpose=project_understanding.get("purpose", "Unknown"),
            tech_stack=tech_stack,
            key_features=key_features,
            architecture=architecture,
            entry_points=entry_points,
            setup_instructions=setup_instructions,
            compressed_overview=compressed_overview
        )
    
    def _analyze_project_type(
        self,
        file_tree: Dict,
        file_contents: Dict,
        statistics: Dict,
        patterns: Dict
    ) -> Dict[str, str]:
        """Understand what type of project this is and its purpose."""
        
        # Prepare context for LLM
        context = f"""Based on the following project information, determine the project type and purpose:

File Statistics:
- Total files: {statistics.get('total_files', 0)}
- Languages: {statistics.get('language_percentages', {})}
- Detected type: {patterns.get('project_type', 'unknown')}
- Framework: {patterns.get('framework', 'none detected')}

README excerpt (if available):
{self._get_readme_excerpt(file_contents)}

Main directory structure:
{self._get_top_level_structure(file_tree)}

Analyze this and provide:
1. Project type (e.g., web app, library, CLI tool, API service)
2. Main purpose in one sentence
"""
        
        try:
            response = self.llm_client.complete(
                prompt=context,
                temperature=0.3,  # Lower temperature for factual analysis
                max_tokens=200
            )
            
            # Parse response
            content = response.content
            
            # Simple extraction (could be enhanced with structured output)
            lines = content.split('\n')
            project_type = "Unknown"
            purpose = "Unknown"
            
            for line in lines:
                if "type:" in line.lower() or "project type:" in line.lower():
                    project_type = line.split(':', 1)[1].strip()
                elif "purpose:" in line.lower() or "main purpose:" in line.lower():
                    purpose = line.split(':', 1)[1].strip()
            
            return {
                "type": project_type,
                "purpose": purpose
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze project type: {e}")
            return {
                "type": patterns.get('project_type', 'Unknown'),
                "purpose": "Could not determine purpose"
            }
    
    def _analyze_tech_stack(
        self,
        file_contents: Dict,
        statistics: Dict,
        patterns: Dict
    ) -> Dict[str, list]:
        """Extract technology stack from dependencies and configs."""
        
        tech_stack = {
            "languages": [],
            "frameworks": [],
            "databases": [],
            "tools": []
        }
        
        # Get top languages
        if statistics.get("languages"):
            tech_stack["languages"] = list(statistics["languages"].keys())[:3]
        
        # Check for specific dependencies
        for filename, content in file_contents.items():
            if "requirements.txt" in filename:
                tech_stack["frameworks"].extend(
                    self._extract_python_deps(content)
                )
            elif "package.json" in filename:
                tech_stack["frameworks"].extend(
                    self._extract_node_deps(content)
                )
        
        # Add detected patterns
        if patterns.get("framework"):
            if patterns["framework"] not in tech_stack["frameworks"]:
                tech_stack["frameworks"].append(patterns["framework"])
        
        if patterns.get("build_tool"):
            tech_stack["tools"].append(patterns["build_tool"])
        
        if patterns.get("containerized"):
            tech_stack["tools"].append("Docker")
        
        return tech_stack
    
    def _extract_key_features(
        self,
        file_contents: Dict,
        file_tree: Dict
    ) -> list:
        """Extract key features from README and structure."""
        
        features = []
        
        # Try to extract from README
        readme_content = self._get_readme_content(file_contents)
        if readme_content:
            prompt = f"""Extract the top 3-5 key features from this README:

{readme_content[:1500]}

List only the feature names, one per line, no explanations."""
            
            try:
                response = self.llm_client.complete(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=100
                )
                
                # Parse features
                lines = response.content.strip().split('\n')
                features = [
                    line.strip('- •*').strip()
                    for line in lines
                    if line.strip() and not line.startswith('#')
                ][:5]
                
            except Exception as e:
                logger.error(f"Failed to extract features: {e}")
        
        # Fallback: infer from structure
        if not features:
            if "api" in str(file_tree).lower():
                features.append("API endpoints")
            if "test" in str(file_tree).lower():
                features.append("Test suite")
            if "docs" in str(file_tree).lower():
                features.append("Documentation")
        
        return features
    
    def _analyze_architecture(
        self,
        file_tree: Dict,
        patterns: Dict,
        statistics: Dict
    ) -> str:
        """Determine architecture pattern."""
        
        # Check directory structure for patterns
        structure_str = json.dumps(file_tree, indent=2)[:1000]
        
        prompt = f"""Based on this project structure, identify the architecture pattern:

Structure:
{structure_str}

Statistics:
- Total files: {statistics.get('total_files', 0)}
- Framework: {patterns.get('framework', 'none')}

Identify the pattern in 2-3 words (e.g., "MVC", "Microservices", "Layered", "Modular")."""
        
        try:
            response = self.llm_client.complete(
                prompt=prompt,
                temperature=0.3,
                max_tokens=50
            )
            
            # Extract pattern
            architecture = response.content.strip().split('\n')[0]
            return architecture
            
        except Exception as e:
            logger.error(f"Failed to analyze architecture: {e}")
            
            # Fallback heuristics
            if "microservice" in structure_str.lower():
                return "Microservices"
            elif "controller" in structure_str.lower() and "model" in structure_str.lower():
                return "MVC"
            elif statistics.get('total_files', 0) < 20:
                return "Simple/Script"
            else:
                return "Modular"
    
    def _identify_entry_points(
        self,
        file_contents: Dict,
        patterns: Dict
    ) -> list:
        """Identify main entry points."""
        
        entry_points = []
        
        # Check common entry point files
        entry_files = ['main.py', 'app.py', 'index.js', 'server.py', 'run.py']
        
        for filename in file_contents:
            base_name = filename.split('/')[-1].lower()
            if base_name in entry_files:
                entry_points.append(filename)
        
        # Check for CLI entry points
        for filename, content in file_contents.items():
            if "if __name__ == '__main__':" in content:
                if filename not in entry_points:
                    entry_points.append(filename)
            elif "def main(" in content or "function main(" in content:
                if filename not in entry_points:
                    entry_points.append(filename)
        
        return entry_points[:3]  # Limit to top 3
    
    def _extract_setup_instructions(
        self,
        file_contents: Dict
    ) -> str:
        """Extract setup instructions from README."""
        
        readme_content = self._get_readme_content(file_contents)
        
        if readme_content:
            # Look for installation/setup section
            prompt = f"""Extract setup/installation instructions from this README:

{readme_content[:2000]}

Provide a brief summary in 2-3 steps. If not found, return "See README for details"."""
            
            try:
                response = self.llm_client.complete(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=150
                )
                
                return response.content.strip()
                
            except Exception as e:
                logger.error(f"Failed to extract setup instructions: {e}")
        
        return "See project documentation for setup instructions"
    
    def _create_compressed_overview(
        self,
        project_understanding: Dict,
        tech_stack: Dict,
        key_features: list,
        architecture: str,
        entry_points: list,
        setup_instructions: str,
        statistics: Dict
    ) -> str:
        """Create final compressed overview under 2000 tokens."""
        
        # Build comprehensive context
        context = f"""Create a concise project overview from this information:

PROJECT TYPE: {project_understanding.get('type', 'Unknown')}
PURPOSE: {project_understanding.get('purpose', 'Unknown')}

TECH STACK:
- Languages: {', '.join(tech_stack.get('languages', []))}
- Frameworks: {', '.join(tech_stack.get('frameworks', []))}
- Tools: {', '.join(tech_stack.get('tools', []))}

KEY FEATURES:
{chr(10).join('- ' + f for f in key_features)}

ARCHITECTURE: {architecture}

ENTRY POINTS:
{chr(10).join('- ' + e for e in entry_points)}

STATISTICS:
- Total files: {statistics.get('total_files', 0)}
- Total size: {statistics.get('total_size_bytes', 0) / 1024 / 1024:.1f} MB

SETUP: {setup_instructions}

Create a comprehensive but concise overview (max 500 words) that would help someone quickly understand this project. Include:
1. What the project does
2. Technology stack
3. Project structure
4. How to get started
5. Key components/modules"""
        
        try:
            response = self.llm_client.complete(
                prompt=context,
                temperature=0.5,
                max_tokens=2000  # Allow full 2000 tokens for overview
            )
            
            overview = response.content
            
            # Add structured summary at the beginning
            structured_summary = f"""## Project Overview

**Type:** {project_understanding.get('type', 'Unknown')}
**Purpose:** {project_understanding.get('purpose', 'Unknown')}
**Architecture:** {architecture}
**Main Language:** {tech_stack.get('languages', ['Unknown'])[0]}

---

{overview}"""
            
            return structured_summary
            
        except Exception as e:
            logger.error(f"Failed to create compressed overview: {e}")
            
            # Fallback to basic summary
            return f"""## Project Overview

**Type:** {project_understanding.get('type', 'Unknown')}
**Purpose:** {project_understanding.get('purpose', 'Unknown')}
**Architecture:** {architecture}
**Languages:** {', '.join(tech_stack.get('languages', []))}
**Frameworks:** {', '.join(tech_stack.get('frameworks', []))}

**Key Features:**
{chr(10).join('- ' + f for f in key_features)}

**Entry Points:** {', '.join(entry_points)}

**Setup:** {setup_instructions}

This project contains {statistics.get('total_files', 0)} files organized in a {architecture} architecture pattern."""
    
    # Helper methods
    
    def _get_readme_excerpt(self, file_contents: Dict) -> str:
        """Get README excerpt if available."""
        for filename, content in file_contents.items():
            if 'readme' in filename.lower():
                return content[:500] + "..." if len(content) > 500 else content
        return "No README found"
    
    def _get_readme_content(self, file_contents: Dict) -> Optional[str]:
        """Get full README content if available."""
        for filename, content in file_contents.items():
            if 'readme' in filename.lower():
                return content
        return None
    
    def _get_top_level_structure(self, file_tree: Dict) -> str:
        """Get top-level directory structure."""
        top_level = []
        for key in list(file_tree.keys())[:10]:  # First 10 items
            if key.endswith('/'):
                top_level.append(f"📁 {key}")
            else:
                top_level.append(f"📄 {key}")
        return '\n'.join(top_level)
    
    def _extract_python_deps(self, requirements_content: str) -> list:
        """Extract key Python dependencies."""
        deps = []
        important_packages = [
            'django', 'flask', 'fastapi', 'pytest', 'numpy',
            'pandas', 'requests', 'sqlalchemy', 'celery'
        ]
        
        for line in requirements_content.split('\n')[:20]:  # Check first 20 lines
            line = line.strip().lower()
            for pkg in important_packages:
                if pkg in line:
                    deps.append(pkg.capitalize())
                    break
        
        return deps[:5]  # Limit to 5
    
    def _extract_node_deps(self, package_json_content: str) -> list:
        """Extract key Node.js dependencies."""
        deps = []
        important_packages = [
            'express', 'react', 'vue', 'angular', 'next',
            'nest', 'jest', 'mocha', 'webpack', 'typescript'
        ]
        
        for pkg in important_packages:
            if f'"{pkg}"' in package_json_content.lower():
                deps.append(pkg.capitalize())
        
        return deps[:5]  # Limit to 5