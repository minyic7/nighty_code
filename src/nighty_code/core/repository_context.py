"""
Repository context manager that loads and manages artifacts for LLM access.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RepositoryContext:
    """
    Manages repository artifacts and provides intelligent access for LLM queries.
    
    This class handles:
    - Loading and caching artifacts (identity cards, relationships, graphs)
    - Providing search and query interfaces
    - Managing session state and artifact freshness
    """
    
    repository_path: Path
    artifacts_dir: Path
    auto_refresh: bool = True
    
    # Cached artifacts
    _identity_cards: Dict[str, Any] = field(default_factory=dict, init=False)
    _relationships: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _repository_graph: Dict[str, Any] = field(default_factory=dict, init=False)
    _classifications: Dict[str, Any] = field(default_factory=dict, init=False)
    _entities: List[Dict[str, Any]] = field(default_factory=list, init=False)
    
    # Metadata
    _loaded_at: Optional[datetime] = field(default=None, init=False)
    _artifacts_exist: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Initialize and check for existing artifacts."""
        self.artifacts_dir = Path(self.artifacts_dir)
        self.repository_path = Path(self.repository_path)
        
        # Check if artifacts exist
        self._check_artifacts()
        
        if self._artifacts_exist:
            logger.info(f"Found existing artifacts in {self.artifacts_dir}")
        else:
            logger.warning(f"No artifacts found in {self.artifacts_dir}")
    
    def _check_artifacts(self) -> bool:
        """Check if required artifact files exist."""
        required_files = [
            "identity_cards_all.json",
            "relationships_all.json",
            "repository_graph.json"
        ]
        
        self._artifacts_exist = all(
            (self.artifacts_dir / f).exists() for f in required_files
        )
        return self._artifacts_exist
    
    def load_artifacts(self, force_reload: bool = False) -> bool:
        """
        Load all artifacts into memory.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successfully loaded
        """
        if self._loaded_at and not force_reload:
            logger.debug("Artifacts already loaded")
            return True
        
        if not self._artifacts_exist:
            logger.error("Cannot load artifacts - they don't exist")
            return False
        
        try:
            # Load identity cards
            cards_file = self.artifacts_dir / "identity_cards_all.json"
            if cards_file.exists():
                with open(cards_file, 'r', encoding='utf-8') as f:
                    cards_data = json.load(f)
                    # Handle both formats: array or object with 'identity_cards' key
                    if isinstance(cards_data, list):
                        cards_list = cards_data
                    else:
                        cards_list = cards_data.get('identity_cards', [])
                    
                    # Index by file path for quick lookup
                    for card in cards_list:
                        self._identity_cards[card['file_path']] = card
            
            # Load relationships
            rel_file = self.artifacts_dir / "relationships_all.json"
            if rel_file.exists():
                with open(rel_file, 'r', encoding='utf-8') as f:
                    rel_data = json.load(f)
                    # Handle both formats: array or object with 'relationships' key
                    if isinstance(rel_data, list):
                        self._relationships = rel_data
                    else:
                        self._relationships = rel_data.get('relationships', [])
            
            # Load repository graph
            graph_file = self.artifacts_dir / "repository_graph.json"
            if graph_file.exists():
                with open(graph_file, 'r', encoding='utf-8') as f:
                    self._repository_graph = json.load(f)
            
            # Load classifications if available
            class_file = self.artifacts_dir / "classifications_all_files.json"
            if class_file.exists():
                with open(class_file, 'r', encoding='utf-8') as f:
                    class_data = json.load(f)
                    # Handle both formats: array or object with 'classifications' key
                    if isinstance(class_data, list):
                        class_list = class_data
                    else:
                        class_list = class_data.get('classifications', [])
                    
                    for item in class_list:
                        self._classifications[item['file_path']] = item
            
            # Load entities if available
            entities_file = self.artifacts_dir / "entities_all.json"
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    # Handle both formats: array or object with 'entities' key
                    if isinstance(entities_data, list):
                        self._entities = entities_data
                    else:
                        self._entities = entities_data.get('entities', [])
            
            self._loaded_at = datetime.now()
            logger.info(f"Loaded artifacts: {len(self._identity_cards)} cards, "
                       f"{len(self._relationships)} relationships, "
                       f"{len(self._entities)} entities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            return False
    
    def needs_refresh(self) -> bool:
        """
        Check if artifacts need to be refreshed based on file modifications.
        
        Returns:
            True if source files have been modified since artifacts were created
        """
        if not self._artifacts_exist:
            return True
        
        # Get artifact creation time
        cards_file = self.artifacts_dir / "identity_cards_all.json"
        artifact_mtime = cards_file.stat().st_mtime if cards_file.exists() else 0
        
        # Check if any source files are newer
        for file_path in self._identity_cards.keys():
            full_path = self.repository_path / file_path
            if full_path.exists():
                if full_path.stat().st_mtime > artifact_mtime:
                    return True
        
        return False
    
    def search_identity_cards(
        self,
        keywords: List[str],
        file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search identity cards by keywords and optionally filter by file type.
        
        Args:
            keywords: Keywords to search for in card content
            file_types: Optional list of file types to filter
            
        Returns:
            List of matching identity cards
        """
        if not self._identity_cards:
            self.load_artifacts()
        
        results = []
        keywords_lower = [k.lower() for k in keywords]
        
        for file_path, card in self._identity_cards.items():
            # Filter by file type if specified
            if file_types and card.get('file_type') not in file_types:
                continue
            
            # Search in various fields
            # Handle entities_defined which can be dicts or strings
            entities_text = ''
            entities_defined = card.get('entities_defined', [])
            if entities_defined:
                if isinstance(entities_defined[0], dict):
                    # Entities are dicts with 'name' field
                    entities_text = ' '.join(e.get('name', '') for e in entities_defined)
                else:
                    # Entities are strings
                    entities_text = ' '.join(entities_defined)
            
            searchable_text = ' '.join(filter(None, [
                card.get('file_name', ''),
                card.get('purpose') or '',
                ' '.join(card.get('quick_facts', [])),
                ' '.join(card.get('key_insights', [])),
                card.get('llm_summary') or '',
                entities_text
            ])).lower()
            
            # Check if any keyword matches
            if any(keyword in searchable_text for keyword in keywords_lower):
                results.append(card)
        
        return results
    
    def get_dependencies(
        self,
        file_path: str,
        direction: str = "both"
    ) -> Dict[str, List[str]]:
        """
        Get dependencies for a file.
        
        Args:
            file_path: Path to the file
            direction: "upstream", "downstream", or "both"
            
        Returns:
            Dictionary with upstream and/or downstream dependencies
        """
        if not self._identity_cards:
            self.load_artifacts()
        
        card = self._identity_cards.get(file_path)
        if not card:
            return {"upstream": [], "downstream": []}
        
        result = {}
        
        if direction in ["upstream", "both"]:
            result["upstream"] = card.get("upstream_dependencies", [])
        
        if direction in ["downstream", "both"]:
            result["downstream"] = card.get("downstream_dependencies", [])
        
        return result
    
    def get_related_files(
        self,
        file_path: str,
        max_depth: int = 2
    ) -> Set[str]:
        """
        Get related files by following dependency chains.
        
        Args:
            file_path: Starting file path
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of related file paths
        """
        if not self._identity_cards:
            self.load_artifacts()
        
        related = set()
        to_visit = [(file_path, 0)]
        visited = set()
        
        while to_visit:
            current_file, depth = to_visit.pop(0)
            
            if current_file in visited or depth > max_depth:
                continue
            
            visited.add(current_file)
            related.add(current_file)
            
            # Get dependencies
            deps = self.get_dependencies(current_file, "both")
            
            # Add to visit queue
            for dep in deps.get("upstream", []):
                if dep not in visited:
                    to_visit.append((dep, depth + 1))
            
            for dep in deps.get("downstream", []):
                if dep not in visited:
                    to_visit.append((dep, depth + 1))
        
        return related
    
    def find_entities(
        self,
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities by type or name pattern.
        
        Args:
            entity_type: Type of entity to find (e.g., "class", "function")
            name_pattern: Pattern to match in entity names
            
        Returns:
            List of matching entities
        """
        if not self._entities:
            self.load_artifacts()
        
        results = []
        
        for entity in self._entities:
            # Filter by type
            if entity_type and entity.get('entity_type') != entity_type:
                continue
            
            # Filter by name pattern
            if name_pattern:
                entity_name = entity.get('name', '').lower()
                if name_pattern.lower() not in entity_name:
                    continue
            
            results.append(entity)
        
        return results
    
    def get_repository_structure(self) -> Dict[str, Any]:
        """
        Get a compact representation of repository structure.
        
        Returns:
            Dictionary with file tree and statistics
        """
        structure = {
            "total_files": len(self._identity_cards),
            "file_types": {},
            "frameworks": set(),
            "complexity_distribution": {},
            "file_tree": {}
        }
        
        for file_path, card in self._identity_cards.items():
            # Count file types
            file_type = card.get('file_type', 'unknown')
            structure["file_types"][file_type] = structure["file_types"].get(file_type, 0) + 1
            
            # Collect frameworks
            for framework in card.get('detected_frameworks', []):
                structure["frameworks"].add(framework)
            
            # Count complexity
            complexity = card.get('complexity_level', 'unknown')
            structure["complexity_distribution"][complexity] = \
                structure["complexity_distribution"].get(complexity, 0) + 1
            
            # Build file tree
            parts = Path(file_path).parts
            current = structure["file_tree"]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = {
                "type": file_type,
                "size": card.get('file_size_bytes', 0)
            }
        
        # Convert set to list for JSON serialization
        structure["frameworks"] = list(structure["frameworks"])
        
        return structure
    
    def get_context_for_query(
        self,
        query: str,
        max_files: int = 10
    ) -> Dict[str, Any]:
        """
        Build optimized context for an LLM query.
        
        Args:
            query: The user's query
            max_files: Maximum number of files to include
            
        Returns:
            Context dictionary optimized for LLM consumption
        """
        # Extract keywords from query (simple approach)
        keywords = [w for w in query.lower().split() if len(w) > 3]
        
        # Search for relevant files
        relevant_cards = self.search_identity_cards(keywords)[:max_files]
        
        # Build focused context
        context = {
            "repository_overview": {
                "total_files": len(self._identity_cards),
                "file_types": list(set(c.get('file_type') for c in self._identity_cards.values()))
            },
            "relevant_files": []
        }
        
        for card in relevant_cards:
            file_context = {
                "path": card.get('file_path'),
                "type": card.get('file_type'),
                "purpose": card.get('purpose'),
                "key_facts": card.get('quick_facts', [])[:3],  # Limit facts
                "entities": card.get('entities_defined', [])[:5],  # Limit entities
                "dependencies": {
                    "upstream": len(card.get('upstream_dependencies', [])),
                    "downstream": len(card.get('downstream_dependencies', []))
                }
            }
            context["relevant_files"].append(file_context)
        
        # Add relationship information if relevant
        if any(k in query.lower() for k in ['depend', 'import', 'use', 'call']):
            context["relationships"] = [
                {
                    "source": r.get('source_file'),
                    "target": r.get('target_file'),
                    "type": r.get('relationship_type')
                }
                for r in self._relationships[:20]  # Limit relationships
            ]
        
        return context
    
    def generate_artifacts(self, force: bool = False) -> bool:
        """
        Generate artifacts for the repository if they don't exist or need refresh.
        
        Args:
            force: Force regeneration even if artifacts exist
            
        Returns:
            True if artifacts were generated successfully
        """
        if self._artifacts_exist and not force and not self.needs_refresh():
            logger.info("Artifacts are up to date")
            return True
        
        logger.info("Generating artifacts for repository...")
        
        try:
            from ..extraction import StructuredExtractor, ExtractionConfig
            from .artifact_manager import ArtifactManager
            
            # Configure extraction
            config = ExtractionConfig(
                use_tree_sitter_when_available=True,
                use_llm_fallback=True,
                generate_identity_cards=True
            )
            
            # Extract from repository
            extractor = StructuredExtractor(config)
            response = extractor.extract_from_repository(
                repository_path=self.repository_path,
                max_files=None
            )
            
            # Save artifacts
            artifact_manager = ArtifactManager(self.artifacts_dir.parent)
            
            # Save all artifact types
            # This would need to be implemented based on the response structure
            logger.info(f"Generated artifacts for {response.files_processed} files")
            
            # Reload artifacts
            self._check_artifacts()
            return self.load_artifacts(force_reload=True)
            
        except Exception as e:
            logger.error(f"Failed to generate artifacts: {e}")
            return False