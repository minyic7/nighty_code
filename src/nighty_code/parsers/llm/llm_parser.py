"""
LLM-based parser for extracting structured information from unsupported file types.
Uses LLM with context from identity cards and classifications for accurate extraction.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Type, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .dynamic_models import (
    DynamicEntity,
    DynamicRelationship,
    DynamicFileEntity,
    DynamicExtractionResult,
    DynamicLocation,
    EntityTypeNormalizer,
    FileType
)
from ...llm.anthropic_client import AnthropicClient
from ...identity.schemas import IdentityCard


logger = logging.getLogger(__name__)


class ParserConfig(BaseModel):
    """Configuration for LLM parser."""
    
    use_identity_context: bool = Field(
        True,
        description="Use identity cards as context for better understanding"
    )
    use_classification_context: bool = Field(
        True,
        description="Use file classifications as context"
    )
    max_context_files: int = Field(
        5,
        description="Maximum number of related files to include as context"
    )
    temperature: float = Field(
        0.2,
        description="LLM temperature for more deterministic parsing"
    )
    max_tokens: int = Field(
        2000,
        description="Maximum tokens for LLM response"
    )
    extract_entities: bool = Field(
        True,
        description="Extract entities from the file"
    )
    extract_relationships: bool = Field(
        True,
        description="Extract relationships between entities"
    )
    extract_metrics: bool = Field(
        True,
        description="Calculate file metrics"
    )


class LLMParser:
    """
    LLM-based parser for extracting structured information from files.
    Falls back to this when tree-sitter parsers are not available.
    """
    
    # Supported file extensions for tree-sitter (prefer these)
    TREE_SITTER_SUPPORTED = {
        '.scala', '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c',
        '.rb', '.php', '.cs', '.swift', '.kt'
    }
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize the LLM parser.
        
        Args:
            config: Parser configuration
        """
        self.config = config or ParserConfig()
        self.llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self) -> Optional[AnthropicClient]:
        """Initialize LLM client if API key is available."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return AnthropicClient(api_key)
        
        logger.warning("No Anthropic API key found. LLM parser will not be available.")
        return None
    
    def should_use_llm(self, file_path: Path) -> bool:
        """
        Determine if LLM parser should be used for this file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if LLM parser should be used (unsupported file type)
        """
        # Check if it's a supported tree-sitter extension
        if file_path.suffix.lower() in self.TREE_SITTER_SUPPORTED:
            return False
        
        # Use LLM for all other file types
        return True
    
    def parse_file(
        self,
        file_path: Path,
        context: Optional[Dict[str, Any]] = None
    ) -> DynamicExtractionResult:
        """
        Parse a file using LLM to extract entities and relationships.
        
        Args:
            file_path: Path to the file to parse
            context: Additional context (identity cards, classifications, etc.)
            
        Returns:
            Extraction result with entities and relationships
        """
        start_time = datetime.now()
        
        if not self.llm_client:
            return DynamicExtractionResult(
                success=False,
                errors=["LLM client not initialized. Check API key."],
                parser_used="llm"
            )
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Calculate basic metrics
            metrics = self._calculate_metrics(content)
            
            # Prepare context for LLM
            llm_context = self._prepare_context(file_path, content, context)
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(file_path, content)
            
            # Call LLM for extraction
            response = self.llm_client.create_message(
                model=os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022"),
                messages=[{"role": "user", "content": prompt}],
                system=llm_context,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse LLM response
            file_entity = self._parse_llm_response(response, file_path, metrics)
            
            # Calculate extraction time
            extraction_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return DynamicExtractionResult(
                success=True,
                file_entity=file_entity,
                parser_used="llm",
                extraction_time_ms=extraction_time
            )
            
        except Exception as e:
            logger.error(f"LLM parsing failed for {file_path}: {e}")
            return DynamicExtractionResult(
                success=False,
                errors=[str(e)],
                parser_used="llm",
                extraction_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic file metrics."""
        lines = content.split('\n')
        
        return {
            "line_count": len(lines),
            "size_bytes": len(content.encode('utf-8')),
            "blank_lines": sum(1 for line in lines if not line.strip()),
            "comment_lines": sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*')))
        }
    
    def _prepare_context(
        self,
        file_path: Path,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Prepare context for LLM from identity cards and classifications.
        """
        context_parts = []
        
        # Add file information
        context_parts.append(f"File: {file_path.name}")
        context_parts.append(f"Path: {file_path}")
        context_parts.append(f"Extension: {file_path.suffix}")
        
        # Add identity card context if available
        if context and self.config.use_identity_context:
            if "identity_cards" in context:
                # Find related files
                related_cards = self._find_related_cards(
                    file_path,
                    context["identity_cards"],
                    self.config.max_context_files
                )
                
                if related_cards:
                    context_parts.append("\nRelated files in the repository:")
                    for card in related_cards:
                        context_parts.append(f"  - {card.file_name}: {card.purpose or 'No description'}")
                        if card.upstream_files:
                            context_parts.append(f"    Dependencies: {', '.join(card.upstream_files[:3])}")
        
        # Add classification context if available
        if context and self.config.use_classification_context:
            if "classification" in context:
                classification = context["classification"]
                context_parts.append(f"\nFile classification:")
                context_parts.append(f"  Type: {classification.get('file_type', 'unknown')}")
                context_parts.append(f"  Complexity: {classification.get('complexity', 'unknown')}")
                context_parts.append(f"  Frameworks: {', '.join(classification.get('frameworks', []))}")
        
        # Add parsing instructions
        context_parts.append("\nYou are a code analysis expert. Extract structured information from the provided file.")
        context_parts.append("Focus on identifying:")
        context_parts.append("  1. Key entities (classes, functions, interfaces, etc.)")
        context_parts.append("  2. Relationships between entities")
        context_parts.append("  3. Import/export statements")
        context_parts.append("  4. Main functionality and purpose")
        
        return "\n".join(context_parts)
    
    def _find_related_cards(
        self,
        file_path: Path,
        identity_cards: List[IdentityCard],
        max_cards: int
    ) -> List[IdentityCard]:
        """Find identity cards most related to the current file."""
        # Simple heuristic: files in same directory or mentioned in imports
        related = []
        
        for card in identity_cards:
            # Same directory
            if Path(card.file_path).parent == file_path.parent:
                related.append(card)
            # Or file might import this
            elif file_path.stem in str(card.downstream_files):
                related.append(card)
            
            if len(related) >= max_cards:
                break
        
        return related
    
    def _create_extraction_prompt(self, file_path: Path, content: str) -> str:
        """Create the extraction prompt for LLM."""
        
        # Truncate content if too long
        max_content_length = 10000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n... [truncated]"
        
        prompt = f"""Analyze the following file and extract structured information.

File: {file_path.name}

Content:
```
{content}
```

Extract the following information and return as JSON:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "class|function|interface|trait|object|method|variable|constant|enum|type_alias|import|export|namespace|package|module",
      "qualified_name": "fully.qualified.name if applicable",
      "line_number": 10,
      "visibility": "public|private|protected",
      "extends": "parent class if applicable",
      "implements": ["interface1", "interface2"],
      "parameters": [{{"name": "param1", "type": "string"}}],
      "return_type": "return type if function/method"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity",
      "target": "target entity",
      "type": "uses|calls|extends|implements|imports|instantiates|references",
      "line_number": 20
    }}
  ],
  "imports": ["import statement 1", "import statement 2"],
  "exports": ["exported entity 1", "exported entity 2"],
  "package": "package or namespace",
  "purpose": "Brief one-line description of what this file does",
  "key_features": ["feature 1", "feature 2", "feature 3"]
}}

Important:
- Extract ALL major entities (classes, functions, interfaces, etc.)
- Identify relationships between entities
- Include line numbers where entities are defined
- Be accurate with entity types
- For unknown file types, do your best to identify logical components

Return ONLY valid JSON, no additional text."""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        file_path: Path,
        metrics: Dict[str, Any]
    ) -> DynamicFileEntity:
        """Parse the LLM response into structured models."""
        
        try:
            # Extract content from Anthropic response
            content = ""
            if "content" in response:
                for block in response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
            
            # Parse JSON from response
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                parsed = {}
            
            # Convert to dynamic models
            entities = []
            for entity_data in parsed.get("entities", []):
                # Use dynamic entity - accept any type
                entity = DynamicEntity(
                    name=entity_data.get("name", "unknown"),
                    entity_type=entity_data.get("type", "unknown"),  # Any string is valid
                    qualified_name=entity_data.get("qualified_name"),
                    location=DynamicLocation(
                        line_start=entity_data.get("line_number", 1)
                    ) if entity_data.get("line_number") else None,
                    visibility=entity_data.get("visibility"),
                    extends=entity_data.get("extends"),
                    implements=entity_data.get("implements", []),
                    parameters=entity_data.get("parameters", []),
                    return_type=entity_data.get("return_type"),
                    # Store any extra fields in metadata
                    metadata={k: v for k, v in entity_data.items() 
                             if k not in ["name", "type", "qualified_name", "line_number", 
                                         "visibility", "extends", "implements", "parameters", "return_type"]}
                )
                
                # Optionally normalize and categorize
                entity.entity_type = EntityTypeNormalizer.normalize(entity.entity_type)
                entity.category = EntityTypeNormalizer.categorize(entity.entity_type)
                
                entities.append(entity)
            
            relationships = []
            for rel_data in parsed.get("relationships", []):
                relationship = DynamicRelationship(
                    source=rel_data.get("source", ""),
                    target=rel_data.get("target", ""),
                    relationship_type=rel_data.get("type", "unknown"),  # Any string is valid
                    location=DynamicLocation(
                        line_start=rel_data.get("line_number", 1)
                    ) if rel_data.get("line_number") else None,
                    confidence=0.8,  # LLM-extracted relationships have lower confidence
                    metadata={k: v for k, v in rel_data.items()
                             if k not in ["source", "target", "type", "line_number"]}
                )
                relationships.append(relationship)
            
            # Determine file type
            file_type = self._detect_file_type(file_path)
            
            # Store any extra top-level fields as metadata
            known_fields = {"entities", "relationships", "imports", "exports", "package", 
                          "purpose", "key_features"}
            extra_metadata = {k: v for k, v in parsed.items() if k not in known_fields}
            
            return DynamicFileEntity(
                file_path=str(file_path),
                file_type=file_type.value if isinstance(file_type, FileType) else file_type,
                package=parsed.get("package"),
                imports=parsed.get("imports", []),
                exports=parsed.get("exports", []),
                entities=entities,
                relationships=relationships,
                metrics=metrics,
                metadata=extra_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return minimal file entity
            file_type = self._detect_file_type(file_path)
            return DynamicFileEntity(
                file_path=str(file_path),
                file_type=file_type.value if isinstance(file_type, FileType) else file_type,
                entities=[],
                relationships=[],
                metrics=metrics
            )
    
    def _detect_file_type(self, file_path: Path) -> FileType:
        """Detect file type from extension."""
        ext_map = {
            '.py': FileType.PYTHON,
            '.scala': FileType.SCALA,
            '.java': FileType.JAVA,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.TYPESCRIPT,
            '.go': FileType.GO,
            '.rs': FileType.RUST,
            '.cpp': FileType.CPP,
            '.cs': FileType.CSHARP,
            '.rb': FileType.RUBY,
            '.php': FileType.PHP,
            '.kt': FileType.KOTLIN,
            '.swift': FileType.SWIFT,
            '.sql': FileType.SQL,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.json': FileType.JSON,
            '.xml': FileType.XML,
            '.md': FileType.MARKDOWN,
            '.txt': FileType.TEXT,
            '.conf': FileType.CONFIG,
            '.cfg': FileType.CONFIG,
            '.ini': FileType.CONFIG,
            '.properties': FileType.CONFIG
        }
        
        return ext_map.get(file_path.suffix.lower(), FileType.UNKNOWN)
    
    def extract_to_schema(
        self,
        file_path: Path,
        target_schema: Type[BaseModel],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseModel]:
        """
        Extract information from a file to match a user-defined Pydantic schema.
        
        Args:
            file_path: Path to the file
            target_schema: Pydantic model class to extract to
            context: Additional context
            
        Returns:
            Instance of target_schema with extracted data, or None if failed
        """
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return None
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Prepare context
            llm_context = self._prepare_context(file_path, content, context)
            
            # Create schema extraction prompt
            schema_json = target_schema.schema()
            prompt = f"""Extract information from the following file to match this schema:

Schema:
{json.dumps(schema_json, indent=2)}

File: {file_path.name}

Content:
```
{content[:10000]}
```

Extract information that matches the schema. If a field cannot be determined from the file, use the default value or null.

Return ONLY valid JSON that matches the schema, no additional text."""
            
            # Call LLM
            response = self.llm_client.create_message(
                model=os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022"),
                messages=[{"role": "user", "content": prompt}],
                system=llm_context,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            content = ""
            if "content" in response:
                for block in response["content"]:
                    if block.get("type") == "text":
                        content += block.get("text", "")
            
            # Extract JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Create instance of target schema
                return target_schema(**parsed)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract to schema: {e}")
            return None
