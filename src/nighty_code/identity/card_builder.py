"""
Main identity card builder for generating file identity cards.

This module provides the IdentityCardBuilder that creates simplified
identity cards for files, with clear upstream/downstream file relationships
for optimal LLM consumption.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

from ..core.models import FileClassification
from ..parsers.model.scala_model import (
    ScalaModelExtractor, 
    ScalaEntity, 
    RepositoryRelationshipGraph,
    CrossFileRelationship
)
from .schemas import IdentityCard, EntityInfo

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class IdentityCardBuilder:
    """
    Builds simplified identity cards for files with clear relationships.
    """
    
    def __init__(self, enable_llm_summaries: bool = False):
        """
        Initialize the identity card builder.
        
        Args:
            enable_llm_summaries: If True, generate LLM summaries for cards (requires API key)
        """
        self.scala_extractor = ScalaModelExtractor()
        self.llm_client = None
        
        # Initialize LLM client if enabled and API key is available
        if enable_llm_summaries:
            self._initialize_llm_client()
    
    def build_card_from_classification(
        self,
        classification: FileClassification,
        repository_path: Optional[Path] = None,
        include_relationships: bool = True
    ) -> IdentityCard:
        """
        Build an identity card from a file classification.
        
        Args:
            classification: The file classification
            repository_path: Path to the repository root
            include_relationships: Whether to extract and include relationships
            
        Returns:
            Complete identity card with relationships
        """
        
        # Create base identity card with new schema
        card = IdentityCard.from_classification(
            classification,
            file_path=classification.file_path,
            repository_path=repository_path
        )
        
        # Add code analysis for Scala files
        if classification.file_path.suffix == '.scala' and include_relationships:
            self._add_scala_analysis(card, classification.file_path)
        
        return card
    
    def build_cards_for_repository(
        self,
        repository_path: Path,
        classifications: List[FileClassification]
    ) -> Dict[Path, IdentityCard]:
        """
        Build identity cards for all files in a repository with relationships.
        
        Args:
            repository_path: Path to the repository root
            classifications: List of file classifications
            
        Returns:
            Dictionary mapping file paths to identity cards
        """
        
        cards = {}
        scala_files = []
        
        # First pass: Create basic cards and identify Scala files
        for classification in classifications:
            card = IdentityCard.from_classification(
                classification,
                file_path=classification.file_path,
                repository_path=repository_path
            )
            cards[classification.file_path] = card
            
            if classification.file_path.suffix == '.scala':
                scala_files.append(classification.file_path)
        
        # Second pass: Extract entities and relationships for Scala files
        if scala_files:
            logger.info(f"Extracting relationships for {len(scala_files)} Scala files")
            
            # Extract entities from all files
            file_entities = []
            for file_path in scala_files:
                try:
                    result = self.scala_extractor.extract_from_file(file_path)
                    if result and result.file_entity:
                        file_entities.append(result.file_entity)
                        
                        # Add entities to the card
                        card = cards[file_path]
                        self._add_entities_to_card(card, result.file_entity)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract entities from {file_path}: {e}")
            
            # Build cross-file relationships
            if file_entities:
                repo_graph = self.scala_extractor.analyze_repository_relationships(file_entities)
                
                logger.info(f"Found {len(repo_graph.file_dependencies)} file dependencies")
                
                # Update cards with relationships using new terminology
                self._update_cards_with_relationships(cards, repo_graph, repository_path)
        
        return cards
    
    def _add_scala_analysis(self, card: IdentityCard, file_path: Path) -> None:
        """
        Add Scala-specific analysis to the identity card.
        
        Args:
            card: The identity card to update
            file_path: Path to the Scala file
        """
        
        try:
            # Extract entities from the file
            result = self.scala_extractor.extract_from_file(file_path)
            
            if result and result.file_entity:
                self._add_entities_to_card(card, result.file_entity)
                
                # Add imports
                card.imports = result.file_entity.imports
                
                # Determine what this file exports (public classes/objects)
                exports = []
                for entity in result.file_entity.entities:
                    if entity.entity_type.value in ["class", "object", "trait"]:
                        exports.append(entity.name)
                card.exports = exports
                
                # Add purpose based on main entities
                if exports:
                    main_export = exports[0]
                    entity_type = next(
                        (e.entity_type.value for e in result.file_entity.entities 
                         if e.name == main_export), 
                        "component"
                    )
                    card.purpose = f"Defines {entity_type} {main_export}"
                
        except Exception as e:
            logger.warning(f"Failed to add Scala analysis for {file_path}: {e}")
    
    def _add_entities_to_card(self, card: IdentityCard, file_entity) -> None:
        """
        Add entities from file entity to identity card.
        
        Args:
            card: Identity card to update
            file_entity: ScalaFileEntity with extracted entities
        """
        
        for entity in file_entity.entities:
            # Get line number if available
            line_number = None
            if entity.location:
                line_number = entity.location.line_start
            
            # Add entity to card
            card.add_entity(
                name=entity.name,
                entity_type=entity.entity_type.value,
                qualified_name=entity.qualified_name,
                line_number=line_number
            )
        
        # Add key functionality based on entity types
        entity_types = set(e.entity_type.value for e in file_entity.entities)
        
        if "main_entry" in entity_types:
            card.key_functionality.append("Entry point (main method)")
        
        if "class" in entity_types or "case_class" in entity_types:
            class_count = sum(1 for e in file_entity.entities 
                            if e.entity_type.value in ["class", "case_class"])
            card.key_functionality.append(f"Defines {class_count} class(es)")
        
        if "object" in entity_types:
            object_count = sum(1 for e in file_entity.entities 
                             if e.entity_type.value == "object")
            card.key_functionality.append(f"Defines {object_count} singleton object(s)")
    
    def _update_cards_with_relationships(
        self,
        cards: Dict[Path, IdentityCard],
        repo_graph: RepositoryRelationshipGraph,
        repository_path: Path
    ) -> None:
        """
        Update identity cards with upstream/downstream relationships.
        
        Args:
            cards: Dictionary of identity cards to update
            repo_graph: Repository relationship graph
            repository_path: Repository root path
        """
        
        # Update file dependencies with new terminology
        for source_file, target_files in repo_graph.file_dependencies.items():
            if source_file in cards:
                source_card = cards[source_file]
                
                # Add upstream files (files this file depends on)
                for target_file in target_files:
                    try:
                        relative_path = str(target_file.relative_to(repository_path))
                        relative_path = relative_path.replace('\\', '/')
                        source_card.add_upstream_file(relative_path)
                        
                        # Add downstream relationship to target file
                        if target_file in cards:
                            target_card = cards[target_file]
                            source_relative = str(source_file.relative_to(repository_path))
                            source_relative = source_relative.replace('\\', '/')
                            target_card.add_downstream_file(source_relative)
                            
                    except ValueError:
                        # If not relative to repository, use file name
                        source_card.add_upstream_file(target_file.name)
        
        # Add imports from entities
        for file_path, entities in repo_graph.entities_by_file.items():
            if file_path in cards:
                card = cards[file_path]
                
                # Extract clean import statements
                imports = []
                for entity in entities:
                    if entity.entity_type.value == "import":
                        import_stmt = entity.name
                        if import_stmt.startswith("import "):
                            import_stmt = import_stmt[7:].strip()
                        if import_stmt and import_stmt not in imports:
                            imports.append(import_stmt)
                
                card.imports = sorted(imports)
                
                # Extract exports (public classes/objects/traits)
                exports = []
                for entity in entities:
                    if entity.entity_type.value in ["class", "object", "trait", "case_class"]:
                        exports.append(entity.name)
                
                card.exports = sorted(set(exports))
        
        # Add relationship-based insights to key_functionality
        for file_path, card in cards.items():
            if card.upstream_files:
                card.key_functionality.append(f"Depends on {len(card.upstream_files)} file(s)")
            
            if card.downstream_files:
                card.key_functionality.append(f"Used by {len(card.downstream_files)} file(s)")
            
            # Identify file role based on relationships
            if not card.upstream_files and card.downstream_files:
                card.key_functionality.append("Base/utility module (no dependencies)")
            elif card.upstream_files and not card.downstream_files:
                card.key_functionality.append("Top-level/orchestrator module")
            elif len(card.downstream_files) > 2:
                card.key_functionality.append("Core module (widely used)")
    
    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client for generating summaries."""
        try:
            # Check for Anthropic API key first
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                from ..llm.anthropic_client import AnthropicClient
                self.llm_client = AnthropicClient(api_key)
                self.llm_model = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
                logger.info(f"Initialized Anthropic LLM client with model {self.llm_model}")
                return
            
            # Check for OpenAI API key as fallback
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                logger.warning("OpenAI support not yet implemented, LLM summaries disabled")
                return
            
            logger.info("No API keys found, LLM summary generation disabled")
            
        except ImportError as e:
            logger.warning(f"Failed to import LLM client: {e}")
            self.llm_client = None
    
    def generate_llm_summary(self, card: IdentityCard) -> Optional[str]:
        """
        Generate an LLM summary for an identity card.
        
        Args:
            card: The identity card to generate a summary for
            
        Returns:
            Generated summary text, or None if generation fails
        """
        if not self.llm_client:
            return None
        
        try:
            # Create a contextual prompt from the identity card
            context = self._create_summary_context(card)
            
            # Create the messages for the API
            messages = [
                {
                    "role": "user",
                    "content": "Based on the file information provided, write a 2-3 sentence summary explaining what this file does and its role in the system. Focus on its main purpose and how it relates to other components."
                }
            ]
            
            # Call the LLM API
            response = self.llm_client.create_message(
                model=self.llm_model,
                messages=messages,
                system=context,
                max_tokens=150,
                temperature=0.3
            )
            
            # Extract and format the response
            if response and "content" in response:
                content = ""
                for block in response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
                
                if content.strip():
                    logger.debug(f"Generated LLM summary for {card.file_name}")
                    return content.strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary for {card.file_name}: {e}")
            return None
    
    def _create_summary_context(self, card: IdentityCard) -> str:
        """
        Create context for LLM summary generation from an identity card.
        
        Args:
            card: The identity card
            
        Returns:
            Context string for the LLM
        """
        parts = []
        
        # Basic file information
        parts.append(f"File: {card.file_name}")
        parts.append(f"Path: {card.file_path}")
        parts.append(f"Type: {card.file_type}")
        parts.append(f"Complexity: {card.complexity}")
        
        if card.line_count:
            parts.append(f"Lines: {card.line_count}")
        
        # Dependencies
        if card.upstream_files:
            parts.append(f"\nThis file depends on:")
            for upstream in card.upstream_files[:5]:  # Limit to avoid token overflow
                parts.append(f"  - {Path(upstream).name}")
        
        if card.downstream_files:
            parts.append(f"\nFiles that depend on this:")
            for downstream in card.downstream_files[:5]:
                parts.append(f"  - {Path(downstream).name}")
        
        # Key entities
        if card.file_entities:
            parts.append(f"\nKey entities defined:")
            entity_types = {}
            for entity in card.file_entities:
                entity_types[entity.type] = entity_types.get(entity.type, [])
                entity_types[entity.type].append(entity.name)
            
            for etype, names in list(entity_types.items())[:5]:
                if len(names) <= 3:
                    parts.append(f"  - {etype}: {', '.join(names)}")
                else:
                    parts.append(f"  - {etype}: {', '.join(names[:3])} and {len(names)-3} more")
        
        # Key functionality
        if card.key_functionality:
            parts.append(f"\nKey functionality:")
            for func in card.key_functionality[:5]:
                parts.append(f"  - {func}")
        
        # Imports (top 5)
        if card.imports:
            parts.append(f"\nKey imports:")
            for imp in card.imports[:5]:
                parts.append(f"  - {imp}")
        
        return "\n".join(parts)
    
    def build_cards_with_summaries(
        self,
        repository_path: Path,
        classifications: List[FileClassification],
        batch_size: int = 10
    ) -> Dict[Path, IdentityCard]:
        """
        Build identity cards with LLM-generated summaries.
        
        Args:
            repository_path: Path to the repository root
            classifications: List of file classifications
            batch_size: Number of cards to process before generating summaries
            
        Returns:
            Dictionary mapping file paths to identity cards with summaries
        """
        # First build all cards with relationships
        cards = self.build_cards_for_repository(repository_path, classifications)
        
        # If LLM client is available, generate summaries
        if self.llm_client:
            logger.info(f"Generating LLM summaries for {len(cards)} files")
            
            processed = 0
            for file_path, card in cards.items():
                # Generate summary for the card
                summary = self.generate_llm_summary(card)
                if summary:
                    card.llm_summary = summary
                
                processed += 1
                if processed % batch_size == 0:
                    logger.info(f"Generated summaries for {processed}/{len(cards)} files")
        
        return cards