"""
Structured extraction module that uses both tree-sitter and LLM parsers
to extract information matching user-defined Pydantic schemas.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Type, Union
from datetime import datetime
from pydantic import BaseModel, Field

from ..parsers.llm import LLMParser, ParserConfig
from ..parsers.llm.dynamic_models import (
    DynamicFileEntity,
    DynamicExtractionResult,
    DynamicEntity,
    EntityTypeNormalizer
)
# Keep old models for compatibility
from ..parsers.llm.models import (
    ExtractionRequest,
    ExtractionResponse,
    RepositoryGraphModel
)
from ..parsers.model.scala_model import ScalaModelExtractor
from ..identity.card_builder import IdentityCardBuilder
from ..core.classifier import FileClassifier


logger = logging.getLogger(__name__)


class ExtractionConfig(BaseModel):
    """Configuration for structured extraction."""
    
    use_tree_sitter_when_available: bool = Field(
        True,
        description="Prefer tree-sitter parsers for supported languages"
    )
    use_llm_fallback: bool = Field(
        True,
        description="Use LLM parser for unsupported file types"
    )
    use_identity_context: bool = Field(
        True,
        description="Use identity cards as context for LLM parsing"
    )
    generate_identity_cards: bool = Field(
        True,
        description="Generate identity cards for all files"
    )
    max_files_per_batch: int = Field(
        10,
        description="Maximum files to process in one batch"
    )
    llm_temperature: float = Field(
        0.2,
        description="Temperature for LLM extraction (lower = more deterministic)"
    )
    llm_max_tokens: int = Field(
        2000,
        description="Maximum tokens for LLM response"
    )


class StructuredExtractor:
    """
    Main extractor that intelligently chooses between tree-sitter and LLM parsers
    to extract structured information matching user-defined schemas.
    """
    
    # Map of extensions to tree-sitter parsers
    TREE_SITTER_PARSERS = {
        '.scala': 'scala',
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        # Add more as tree-sitter parsers are implemented
    }
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the structured extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()
        
        # Initialize parsers
        self.llm_parser = LLMParser(
            ParserConfig(
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                use_identity_context=self.config.use_identity_context
            )
        )
        self.scala_extractor = ScalaModelExtractor()
        
        # Initialize support modules
        self.classifier = FileClassifier()
        self.card_builder = IdentityCardBuilder(enable_llm_summaries=False)
        
        # Track statistics
        self.stats = {
            "files_processed": 0,
            "tree_sitter_used": 0,
            "llm_used": 0,
            "failures": 0,
            "total_time_ms": 0
        }
    
    def extract_from_file(
        self,
        file_path: Path,
        target_schema: Optional[Type[BaseModel]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[DynamicExtractionResult, BaseModel, None]:
        """
        Extract structured information from a single file.
        
        Args:
            file_path: Path to the file
            target_schema: Optional Pydantic schema to extract to
            context: Additional context for extraction
            
        Returns:
            Extraction result or instance of target_schema
        """
        start_time = datetime.now()
        
        # Determine which parser to use
        parser_type = self._select_parser(file_path)
        
        try:
            if parser_type == "tree-sitter":
                result = self._extract_with_tree_sitter(file_path)
                self.stats["tree_sitter_used"] += 1
            elif parser_type == "llm":
                result = self._extract_with_llm(file_path, context)
                self.stats["llm_used"] += 1
            else:
                result = DynamicExtractionResult(
                    success=False,
                    errors=[f"No parser available for {file_path.suffix}"],
                    parser_used="none"
                )
                self.stats["failures"] += 1
            
            self.stats["files_processed"] += 1
            self.stats["total_time_ms"] += (datetime.now() - start_time).total_seconds() * 1000
            
            # If target schema provided, extract to that schema
            if target_schema and result.success:
                return self._extract_to_schema(result, target_schema, file_path, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            self.stats["failures"] += 1
            return DynamicExtractionResult(
                success=False,
                errors=[str(e)],
                parser_used=parser_type
            )
    
    def extract_from_repository(
        self,
        repository_path: Path,
        target_schema: Optional[Type[BaseModel]] = None,
        file_pattern: str = "**/*",
        max_files: Optional[int] = None
    ) -> ExtractionResponse:
        """
        Extract structured information from an entire repository.
        
        Args:
            repository_path: Path to the repository
            target_schema: Optional Pydantic schema for extraction
            file_pattern: Glob pattern for files to process
            max_files: Maximum number of files to process
            
        Returns:
            Extraction response with all results
        """
        start_time = datetime.now()
        
        # Find files to process
        files = list(repository_path.glob(file_pattern))
        if max_files:
            files = files[:max_files]
        
        # Filter to actual files (not directories)
        files = [f for f in files if f.is_file()]
        
        logger.info(f"Processing {len(files)} files from {repository_path}")
        
        # Generate identity cards if configured
        identity_cards = {}
        if self.config.generate_identity_cards:
            identity_cards = self._generate_identity_cards(repository_path, files)
        
        # Process files
        file_results = []
        extracted_data = []
        context = {
            "identity_cards": list(identity_cards.values()),
            "repository_path": repository_path
        }
        
        for file_path in files:
            # Add classification to context
            try:
                classification = self.classifier.classify_file(file_path)
                context["classification"] = {
                    "file_type": classification.classification_result.file_type.value,
                    "complexity": classification.complexity_level.value,
                    "frameworks": [f.value for f in classification.classification_result.frameworks]
                }
            except:
                pass
            
            # Extract from file
            result = self.extract_from_file(file_path, target_schema, context)
            
            if isinstance(result, DynamicExtractionResult):
                file_results.append(result)
                if result.success and target_schema:
                    # Try to extract to schema
                    schema_instance = self._extract_to_schema(
                        result, target_schema, file_path, context
                    )
                    if schema_instance:
                        # Use model_dump for Pydantic v2
                        extracted_data.append(schema_instance.model_dump() if hasattr(schema_instance, 'model_dump') else schema_instance.dict())
            elif isinstance(result, BaseModel):
                # Direct schema extraction
                extracted_data.append(result.model_dump() if hasattr(result, 'model_dump') else result.dict())
                # Create a success result for tracking
                file_results.append(DynamicExtractionResult(
                    success=True,
                    parser_used="llm" if self.llm_parser.should_use_llm(file_path) else "tree-sitter"
                ))
        
        # Build repository graph if we have results
        repo_graph = None
        if file_results and any(r.success for r in file_results):
            repo_graph = self._build_repository_graph(file_results)
        
        # Prepare final extracted data
        final_extracted = None
        if target_schema and extracted_data:
            # If extracting to a specific schema, wrap list in dict
            final_extracted = {"extracted_schemas": extracted_data}
        elif not target_schema and file_results:
            # Return raw extraction results
            final_extracted = {
                "entities": [],
                "relationships": [],
                "files": []
            }
            for result in file_results:
                if result.success and result.file_entity:
                    final_extracted["files"].append({
                        "path": result.file_entity.file_path,
                        "type": result.file_entity.file_type,
                        "entities": len(result.file_entity.entities),
                        "relationships": len(result.file_entity.relationships)
                    })
                    # Handle dynamic entities which might be dicts or objects
                    for entity in result.file_entity.entities:
                        if isinstance(entity, dict):
                            final_extracted["entities"].append(entity)
                        else:
                            final_extracted["entities"].append(
                                entity.model_dump() if hasattr(entity, 'model_dump') else entity.dict()
                            )
                    for rel in result.file_entity.relationships:
                        if isinstance(rel, dict):
                            final_extracted["relationships"].append(rel)
                        else:
                            final_extracted["relationships"].append(
                                rel.model_dump() if hasattr(rel, 'model_dump') else rel.dict()
                            )
        
        # Convert DynamicExtractionResult to dict for compatibility
        converted_results = []
        for result in file_results:
            if hasattr(result, 'model_dump'):
                converted_results.append(result.model_dump())
            elif hasattr(result, 'dict'):
                converted_results.append(result.dict())
            else:
                # Convert manually if needed
                converted_results.append({
                    "success": result.success,
                    "parser_used": result.parser_used,
                    "errors": result.errors if hasattr(result, 'errors') else [],
                    "warnings": result.warnings if hasattr(result, 'warnings') else []
                })
        
        return ExtractionResponse(
            success=len([r for r in file_results if r.success]) > 0,
            files_processed=len(files),
            files_failed=len([r for r in file_results if not r.success]),
            extracted_data=final_extracted,
            file_results=converted_results,
            repository_graph=repo_graph,
            total_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            llm_tokens_used=self.stats.get("llm_tokens_used", 0)
        )
    
    def _select_parser(self, file_path: Path) -> str:
        """
        Select the appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser type: "tree-sitter", "llm", or "none"
        """
        # Check if tree-sitter is available and preferred
        if self.config.use_tree_sitter_when_available:
            if file_path.suffix.lower() in self.TREE_SITTER_PARSERS:
                return "tree-sitter"
        
        # Check if LLM fallback is enabled
        if self.config.use_llm_fallback:
            if self.llm_parser.llm_client:
                return "llm"
        
        # No parser available
        return "none"
    
    def _extract_with_tree_sitter(self, file_path: Path) -> DynamicExtractionResult:
        """Extract using tree-sitter parser."""
        
        # Currently only Scala is implemented
        if file_path.suffix.lower() == '.scala':
            try:
                result = self.scala_extractor.extract_from_file(file_path)
                
                if result and result.file_entity:
                    # Convert to dynamic model
                    entities = []
                    for entity in result.file_entity.entities:
                        # Convert ScalaEntity to DynamicEntity
                        from ..parsers.llm.dynamic_models import DynamicLocation
                        
                        entity_dict = {
                            "name": entity.name,
                            "entity_type": entity.entity_type.value,
                            "qualified_name": entity.qualified_name,
                            "location": DynamicLocation(
                                line_start=entity.location.line_start,
                                line_end=entity.location.line_end
                            ) if entity.location else None,
                            "category": EntityTypeNormalizer.categorize(entity.entity_type.value)
                        }
                        entities.append(DynamicEntity(**entity_dict))
                    
                    # Create DynamicFileEntity
                    file_entity = DynamicFileEntity(
                        file_path=str(file_path),
                        file_type="scala",
                        imports=result.file_entity.imports,
                        entities=entities,
                        relationships=[],  # TODO: Convert relationships
                        metrics={
                            "line_count": file_path.read_text().count('\n') + 1,
                            "size_bytes": file_path.stat().st_size
                        }
                    )
                    
                    return DynamicExtractionResult(
                        success=True,
                        file_entity=file_entity,
                        parser_used="tree-sitter"
                    )
            except Exception as e:
                logger.error(f"Tree-sitter extraction failed: {e}")
        
        return DynamicExtractionResult(
            success=False,
            errors=["Tree-sitter parser not implemented for this file type"],
            parser_used="tree-sitter"
        )
    
    def _extract_with_llm(
        self,
        file_path: Path,
        context: Optional[Dict[str, Any]] = None
    ) -> DynamicExtractionResult:
        """Extract using LLM parser."""
        return self.llm_parser.parse_file(file_path, context)
    
    def _extract_to_schema(
        self,
        result: DynamicExtractionResult,
        target_schema: Type[BaseModel],
        file_path: Path,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseModel]:
        """
        Extract information to match a user-defined schema.
        
        Args:
            result: Extraction result with entities and relationships
            target_schema: Target Pydantic schema
            file_path: Path to the file
            context: Additional context
            
        Returns:
            Instance of target_schema or None
        """
        # Use LLM to extract to the specific schema
        return self.llm_parser.extract_to_schema(file_path, target_schema, context)
    
    def _generate_identity_cards(
        self,
        repository_path: Path,
        files: List[Path]
    ) -> Dict[Path, Any]:
        """Generate identity cards for files."""
        
        logger.info("Generating identity cards for context...")
        
        # Classify files
        classifications = []
        for file_path in files:
            try:
                classification = self.classifier.classify_file(file_path)
                classifications.append(classification)
            except:
                pass
        
        # Build identity cards
        if classifications:
            cards = self.card_builder.build_cards_for_repository(
                repository_path,
                classifications
            )
            return cards
        
        return {}
    
    def _build_repository_graph(
        self,
        file_results: List[DynamicExtractionResult]
    ) -> RepositoryGraphModel:
        """Build repository-level dependency graph."""
        
        # Count totals
        total_entities = 0
        file_deps = {}
        
        for result in file_results:
            if result.success and result.file_entity:
                total_entities += len(result.file_entity.entities)
                
                # Simple dependency detection based on imports
                file_path = Path(result.file_entity.file_path)
                deps = []
                for imp in result.file_entity.imports:
                    # Try to resolve import to file
                    # This is simplified - real implementation would be more sophisticated
                    if '.' in imp:
                        potential_file = imp.split('.')[-1] + '.scala'
                        deps.append(potential_file)
                
                if deps:
                    file_deps[str(file_path)] = deps
        
        return RepositoryGraphModel(
            total_files=len(file_results),
            total_entities=total_entities,
            file_dependencies=file_deps,
            cross_file_relationships=[],  # Would need more analysis
            entry_points=[],  # Would need to identify main methods
            orphan_files=[]  # Would need to identify isolated files
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            **self.stats,
            "success_rate": (
                (self.stats["files_processed"] - self.stats["failures"]) / 
                self.stats["files_processed"] * 100
            ) if self.stats["files_processed"] > 0 else 0
        }