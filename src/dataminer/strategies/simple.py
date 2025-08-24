# src/dataminer/strategies/simple.py
"""Simple single-pass extraction strategy"""

from typing import Dict, List, Optional, Any, Union, Type
import json
import logging
from datetime import datetime

from .base import BaseExtractionStrategy, ExtractionContext, T
from ..core.types import (
    ExtractionRequest, ExtractionResult, ExtractionConfig,
    ProcessingMode, ExtractionStage
)
from ..core.exceptions import ExtractionError, ValidationError
from ..models.base import ExtractionSchema

logger = logging.getLogger(__name__)


class SimpleExtractionStrategy(BaseExtractionStrategy[T]):
    """Simple single-pass extraction strategy for fast processing"""
    
    def __init__(self):
        super().__init__("SimpleExtraction")
    
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Supports FAST and HYBRID modes"""
        return mode in [ProcessingMode.FAST, ProcessingMode.HYBRID]
    
    async def extract(
        self,
        request: ExtractionRequest[T],
        config: ExtractionConfig
    ) -> ExtractionResult[T]:
        """Execute simple single-pass extraction"""
        
        # Create extraction context
        context = ExtractionContext(request=request, config=config)
        
        # Validate request
        validation_issues = await self._validate_request(request)
        if validation_issues:
            context.add_error(f"Request validation failed: {', '.join(validation_issues)}")
            return await self._create_result(context, None)
        
        # Initialize integrations
        context = await self._initialize_integrations(context)
        
        if not context.llm_client:
            context.add_error("LLM client not available")
            return await self._create_result(context, None)
        
        try:
            # Single extraction stage
            stage = ExtractionStage.INITIAL
            progress = context.get_stage_progress(stage)
            progress.start()
            await self._notify_stage_started(stage, progress)
            
            # Perform extraction
            result_data = await self._perform_single_extraction(context)
            
            if result_data:
                progress.complete()
                context.set_stage_confidence(stage, 0.8)  # Good confidence for simple extraction
            else:
                progress.fail("Extraction failed")
                context.add_error("Failed to extract data")
            
            await self._notify_stage_completed(stage, progress)
            
            return await self._create_result(context, result_data)
            
        except Exception as e:
            context.add_error(f"Extraction failed: {str(e)}")
            self.logger.error(f"Simple extraction error: {e}", exc_info=True)
            return await self._create_result(context, None)
    
    async def _perform_single_extraction(self, context: ExtractionContext) -> Optional[T]:
        """Perform single-pass extraction using LLM"""
        try:
            # Prepare content
            if isinstance(context.request.content, list):
                content = "\n\n---\n\n".join(str(item) for item in context.request.content)
            else:
                content = str(context.request.content)
            
            # Track source
            context.sources_processed.append("input_content")
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(context.request.schema_model, content)
            
            # Prepare LLM request
            from src.llm import Message, MessageRole
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are an expert data extraction assistant. Extract structured data from the provided content according to the specified schema. Be accurate and comprehensive."
                ),
                Message(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]
            
            # Call LLM with structured output
            self.logger.info("Calling LLM for extraction")
            response = await context.llm_client.complete(
                messages=messages,
                response_model=context.request.schema_model,
                temperature=context.request.temperature,
                max_tokens=context.request.max_tokens
            )
            
            # Store raw extraction
            if hasattr(response, 'model_dump'):
                context.raw_extractions.append(response.model_dump())
            else:
                context.raw_extractions.append(response.__dict__)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Single extraction failed: {e}")
            context.add_error(f"LLM extraction failed: {str(e)}")
            return None
    
    def _create_extraction_prompt(self, schema_model: Type[T], content: str) -> str:
        """Create extraction prompt for the schema"""
        
        # Get schema information
        schema_name = schema_model.__name__
        
        # Generate field descriptions - handle both Pydantic v1 and v2
        field_descriptions = []
        
        # Try Pydantic v2 first (model_fields)
        if hasattr(schema_model, 'model_fields'):
            for field_name, field_info in schema_model.model_fields.items():
                # Pydantic v2 field info
                description = field_info.description or field_name
                field_type = str(field_info.annotation) if field_info.annotation else 'Any'
                field_descriptions.append(f"- {field_name} ({field_type}): {description}")
        # Fall back to Pydantic v1 (__fields__)
        elif hasattr(schema_model, '__fields__'):
            for field_name, field_info in schema_model.__fields__.items():
                # Pydantic v1 field info - safely get attributes
                description = field_name  # Default to field name
                field_type = 'Any'  # Default type
                
                # Try to get description
                if hasattr(field_info, 'field_info') and hasattr(field_info.field_info, 'description'):
                    description = field_info.field_info.description or field_name
                elif hasattr(field_info, 'description'):
                    description = field_info.description or field_name
                
                # Try to get type annotation
                if hasattr(field_info, 'annotation'):
                    field_type = str(field_info.annotation)
                elif hasattr(field_info, 'type_'):
                    field_type = str(field_info.type_)
                
                field_descriptions.append(f"- {field_name} ({field_type}): {description}")
        
        field_desc_text = "\n".join(field_descriptions) if field_descriptions else "See schema definition for field details."
        
        prompt = f"""
Extract structured data from the following content according to the {schema_name} schema.

Schema Fields:
{field_desc_text}

Content to analyze:
{content}

Instructions:
1. Extract all relevant information that matches the schema fields
2. Be comprehensive but accurate - don't make up information
3. If certain fields cannot be determined from the content, leave them as null/empty
4. Pay attention to data types and format requirements
5. Ensure the extracted data is well-structured and complete

Please extract the data now:
"""
        return prompt.strip()
    
    def _estimate_extraction_time(self, content_length: int) -> float:
        """Estimate extraction time for simple strategy"""
        # Simple linear estimation: ~1 second per 1000 characters
        return max(5.0, content_length / 1000.0)