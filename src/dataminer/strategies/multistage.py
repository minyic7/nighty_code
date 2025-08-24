# src/dataminer/strategies/multistage.py
"""Multi-stage extraction strategy with refinement and validation"""

from typing import Dict, List, Optional, Any, Union, Type, Callable
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass

from .base import BaseExtractionStrategy, ExtractionContext, T
from ..core.types import (
    ExtractionRequest, ExtractionResult, ExtractionConfig,
    ProcessingMode, ExtractionStage
)
from ..core.exceptions import ExtractionError, ValidationError
from ..models.base import ExtractionSchema


@dataclass
class StageResult:
    """Result of a single extraction stage"""
    stage: ExtractionStage
    success: bool
    data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    errors: List[str] = None
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class StageProcessor:
    """Processor for individual extraction stages"""
    
    def __init__(self, context: ExtractionContext):
        self.context = context
        self.logger = context.request.request_id  # Use request ID for logging context
    
    def _chunk_content(self, content: str, chunk_size: int, overlap: int = 200) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    async def process_discovery(self) -> StageResult:
        """Discovery stage - find and prepare relevant content"""
        start_time = datetime.now()
        
        try:
            discovered_content = []
            
            # Process content based on type
            if isinstance(self.context.request.content, list):
                # Multiple content items
                for i, content_item in enumerate(self.context.request.content):
                    processed = await self._process_content_item(content_item, f"item_{i}")
                    if processed:
                        discovered_content.extend(processed)
            else:
                # Single content item
                processed = await self._process_content_item(self.context.request.content, "main_content")
                if processed:
                    discovered_content.extend(processed)
            
            # Use MCP tools if available for file-based discovery
            if self.context.mcp_server and self.context.request.file_patterns:
                mcp_content = await self._discover_with_mcp()
                discovered_content.extend(mcp_content)
            
            self.context.discovered_content = discovered_content
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = 1.0 if discovered_content else 0.0
            
            return StageResult(
                stage=ExtractionStage.DISCOVERY,
                success=bool(discovered_content),
                data={"content_items": len(discovered_content)},
                confidence=confidence,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=ExtractionStage.DISCOVERY,
                success=False,
                confidence=0.0,
                errors=[f"Discovery failed: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    async def process_initial_extraction(self) -> StageResult:
        """Initial extraction stage - first pass data extraction"""
        start_time = datetime.now()
        
        try:
            if not self.context.discovered_content:
                return StageResult(
                    stage=ExtractionStage.INITIAL,
                    success=False,
                    confidence=0.0,
                    errors=["No content discovered for extraction"]
                )
            
            # Combine discovered content
            combined_content = "\n\n---SECTION BREAK---\n\n".join(self.context.discovered_content)
            
            # Chunk content if too large
            chunks = self._chunk_content(
                combined_content,
                self.context.config.chunk_size,
                self.context.config.overlap_size
            )
            
            # Process chunks
            chunk_results = []
            for i, chunk in enumerate(chunks):
                result = await self._extract_from_chunk(chunk, f"chunk_{i}")
                if result:
                    chunk_results.append(result)
                    self.context.raw_extractions.append(result)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = len(chunk_results) / len(chunks) if chunks else 0.0
            
            return StageResult(
                stage=ExtractionStage.INITIAL,
                success=bool(chunk_results),
                data={"chunks_processed": len(chunks), "successful_extractions": len(chunk_results)},
                confidence=confidence,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=ExtractionStage.INITIAL,
                success=False,
                confidence=0.0,
                errors=[f"Initial extraction failed: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    async def process_refinement(self) -> StageResult:
        """Refinement stage - improve and consolidate extractions"""
        start_time = datetime.now()
        
        try:
            if not self.context.raw_extractions:
                return StageResult(
                    stage=ExtractionStage.REFINEMENT,
                    success=False,
                    confidence=0.0,
                    errors=["No raw extractions to refine"]
                )
            
            # Consolidate multiple extractions
            refined_data = await self._consolidate_extractions(self.context.raw_extractions)
            
            # Improve data quality
            improved_data = await self._improve_extraction_quality(refined_data)
            
            self.context.refined_data = improved_data
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = 0.8  # High confidence after refinement
            
            return StageResult(
                stage=ExtractionStage.REFINEMENT,
                success=bool(improved_data),
                data=improved_data,
                confidence=confidence,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=ExtractionStage.REFINEMENT,
                success=False,
                confidence=0.0,
                errors=[f"Refinement failed: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    async def process_validation(self) -> StageResult:
        """Validation stage - validate against schema and improve quality"""
        start_time = datetime.now()
        
        try:
            data_to_validate = self.context.refined_data or (
                self.context.raw_extractions[-1] if self.context.raw_extractions else None
            )
            
            if not data_to_validate:
                return StageResult(
                    stage=ExtractionStage.VALIDATION,
                    success=False,
                    confidence=0.0,
                    errors=["No data to validate"]
                )
            
            # Validate against schema
            validated_data, validation_errors = await self._validate_data(data_to_validate)
            
            # Store validation results
            self.context.validation_results.append({
                "data": validated_data,
                "errors": validation_errors,
                "timestamp": datetime.now()
            })
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = 0.9 if not validation_errors else max(0.3, 0.9 - len(validation_errors) * 0.1)
            
            return StageResult(
                stage=ExtractionStage.VALIDATION,
                success=bool(validated_data),
                data=validated_data,
                confidence=confidence,
                errors=validation_errors,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return StageResult(
                stage=ExtractionStage.VALIDATION,
                success=False,
                confidence=0.0,
                errors=[f"Validation failed: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    async def _process_content_item(self, content: Any, item_id: str) -> List[str]:
        """Process a single content item"""
        if isinstance(content, str):
            return [content] if content.strip() else []
        elif isinstance(content, dict):
            # Extract text content from dict
            text_parts = []
            for key, value in content.items():
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
            return ["\n".join(text_parts)] if text_parts else []
        else:
            return [str(content)] if content else []
    
    async def _discover_with_mcp(self) -> List[str]:
        """Use MCP tools to discover additional content"""
        discovered = []
        
        try:
            from src.mcp.core.types import ToolCall as MCPToolCall
            
            # Search for files matching patterns
            for pattern in self.context.request.file_patterns or []:
                tool_call = MCPToolCall(
                    name="search_pattern",
                    arguments={"pattern": pattern}
                )
                
                result = await self.context.mcp_server.call_tool(tool_call)
                
                if result.status == "success" and result.content:
                    file_list = result.content[0].text
                    # Read each file
                    for file_path in file_list.split('\n'):
                        if file_path.strip():
                            file_content = await self._read_file_with_mcp(file_path.strip())
                            if file_content:
                                discovered.append(file_content)
                                self.context.sources_processed.append(file_path.strip())
        
        except Exception as e:
            self.context.add_warning(f"MCP discovery failed: {e}")
        
        return discovered
    
    async def _read_file_with_mcp(self, file_path: str) -> Optional[str]:
        """Read a file using MCP tools"""
        try:
            from src.mcp.core.types import ToolCall as MCPToolCall
            
            tool_call = MCPToolCall(
                name="read_file",
                arguments={"path": file_path}
            )
            
            result = await self.context.mcp_server.call_tool(tool_call)
            
            if result.status == "success" and result.content:
                return result.content[0].text
        
        except Exception:
            pass  # Silently fail for individual files
        
        return None
    
    async def _extract_from_chunk(self, content: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Extract data from a content chunk"""
        try:
            from src.llm import Message, MessageRole
            
            prompt = self._create_extraction_prompt(content, chunk_id)
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="Extract structured data from the provided content. Be accurate and comprehensive."
                ),
                Message(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]
            
            response = await self.context.llm_client.complete(
                messages=messages,
                response_model=self.context.request.schema_model,
                temperature=self.context.request.temperature
            )
            
            return response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            
        except Exception as e:
            self.context.add_warning(f"Chunk extraction failed for {chunk_id}: {e}")
            return None
    
    async def _consolidate_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate multiple extractions into one"""
        if not extractions:
            return {}
        
        if len(extractions) == 1:
            return extractions[0]
        
        # Use LLM to consolidate multiple extractions
        try:
            from src.llm import Message, MessageRole
            
            consolidation_prompt = f"""
Consolidate the following {len(extractions)} data extractions into a single, comprehensive result.

Extractions to consolidate:
{json.dumps(extractions, indent=2)}

Instructions:
1. Merge complementary information from all extractions
2. Resolve conflicts by preferring more complete/detailed information
3. Eliminate duplicates while preserving all unique information
4. Ensure the result follows the target schema structure
5. Maintain high data quality and accuracy

Please provide the consolidated result:
"""
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are an expert at consolidating extracted data while maintaining accuracy and completeness."
                ),
                Message(
                    role=MessageRole.USER,
                    content=consolidation_prompt
                )
            ]
            
            response = await self.context.llm_client.complete(
                messages=messages,
                response_model=self.context.request.schema_model,
                temperature=0.1  # Low temperature for consistency
            )
            
            return response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            
        except Exception as e:
            self.context.add_warning(f"Consolidation failed, using first extraction: {e}")
            return extractions[0]
    
    async def _improve_extraction_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Improve the quality of extracted data"""
        try:
            from src.llm import Message, MessageRole
            
            improvement_prompt = f"""
Review and improve the quality of this extracted data:

Current data:
{json.dumps(data, indent=2)}

Improvements to make:
1. Fill in any missing information that can be inferred
2. Standardize formats and naming conventions
3. Add relevant details that might have been missed
4. Ensure all fields are properly populated
5. Validate data consistency and accuracy

Please provide the improved data:
"""
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a data quality expert. Improve extracted data while maintaining accuracy."
                ),
                Message(
                    role=MessageRole.USER,
                    content=improvement_prompt
                )
            ]
            
            response = await self.context.llm_client.complete(
                messages=messages,
                response_model=self.context.request.schema_model,
                temperature=0.2
            )
            
            return response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            
        except Exception as e:
            self.context.add_warning(f"Quality improvement failed: {e}")
            return data
    
    async def _validate_data(self, data: Dict[str, Any]) -> tuple[Optional[T], List[str]]:
        """Validate data against schema"""
        errors = []
        
        try:
            # Try to create instance from data
            validated_instance = self.context.request.schema_model(**data)
            
            # Perform completeness validation
            completeness_result = validated_instance.validate_completeness()
            
            # Add any missing field warnings
            if completeness_result.get("missing_required"):
                errors.extend([f"Missing required field: {field}" 
                             for field in completeness_result["missing_required"]])
            
            return validated_instance, errors
            
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
            return None, errors
    
    def _create_extraction_prompt(self, content: str, context_id: str) -> str:
        """Create extraction prompt for content"""
        schema_name = self.context.request.schema_model.__name__
        
        return f"""
Extract data from the following content according to the {schema_name} schema.

Content ({context_id}):
{content}

Instructions:
1. Extract all relevant information that matches the schema
2. Be thorough and accurate
3. If information is not available, leave fields empty rather than guessing
4. Pay attention to data types and format requirements

Extract the data now:
"""


class MultiStageExtractionStrategy(BaseExtractionStrategy[T]):
    """Multi-stage extraction with discovery, extraction, refinement, and validation"""
    
    def __init__(self):
        super().__init__("MultiStageExtraction")
    
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Supports THOROUGH and HYBRID modes"""
        return mode in [ProcessingMode.THOROUGH, ProcessingMode.HYBRID]
    
    async def extract(
        self,
        request: ExtractionRequest[T],
        config: ExtractionConfig
    ) -> ExtractionResult[T]:
        """Execute multi-stage extraction process"""
        
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
        
        # Create stage processor
        processor = StageProcessor(context)
        
        # Execute enabled stages
        final_result = None
        
        try:
            for stage in request.enabled_stages:
                # Update progress
                progress = context.get_stage_progress(stage)
                progress.start()
                await self._notify_stage_started(stage, progress)
                
                # Execute stage
                stage_result = await self._execute_stage(processor, stage)
                
                # Update context with results
                context.set_stage_confidence(stage, stage_result.confidence)
                
                if stage_result.success:
                    progress.complete()
                    if stage_result.data:
                        final_result = stage_result.data
                else:
                    progress.fail(f"Stage failed: {', '.join(stage_result.errors)}")
                    for error in stage_result.errors:
                        context.add_error(error, stage)
                    
                    # Decide whether to continue or abort
                    if stage in [ExtractionStage.DISCOVERY, ExtractionStage.INITIAL]:
                        # Critical stages - abort if failed
                        break
                
                await self._notify_stage_completed(stage, progress)
            
            # Convert final result to schema instance
            final_instance = None
            if final_result and isinstance(final_result, dict):
                try:
                    final_instance = request.schema_model(**final_result)
                except Exception as e:
                    context.add_warning(f"Failed to create final schema instance: {e}")
            elif final_result and hasattr(final_result, '__dict__'):
                final_instance = final_result
            
            return await self._create_result(context, final_instance)
            
        except Exception as e:
            context.add_error(f"Multi-stage extraction failed: {str(e)}")
            return await self._create_result(context, None)
    
    async def _execute_stage(self, processor: StageProcessor, stage: ExtractionStage) -> StageResult:
        """Execute a specific extraction stage"""
        
        stage_methods = {
            ExtractionStage.DISCOVERY: processor.process_discovery,
            ExtractionStage.INITIAL: processor.process_initial_extraction,
            ExtractionStage.REFINEMENT: processor.process_refinement,
            ExtractionStage.VALIDATION: processor.process_validation,
        }
        
        method = stage_methods.get(stage)
        if not method:
            return StageResult(
                stage=stage,
                success=False,
                confidence=0.0,
                errors=[f"Unsupported stage: {stage}"]
            )
        
        try:
            # Execute with timeout
            timeout = processor.context.config.max_processing_time_seconds
            return await self._with_timeout(
                method(),
                timeout,
                f"stage_{stage.value}"
            )
        except Exception as e:
            return StageResult(
                stage=stage,
                success=False,
                confidence=0.0,
                errors=[f"Stage execution failed: {str(e)}"]
            )