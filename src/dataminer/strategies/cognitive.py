# src/dataminer/strategies/cognitive.py
"""Cognitive extraction strategy using Copilot reasoning"""

from typing import Dict, List, Optional, Any, Union, Type
import json
from datetime import datetime
from dataclasses import dataclass

from .base import BaseExtractionStrategy, ExtractionContext, T
from ..core.types import (
    ExtractionRequest, ExtractionResult, ExtractionConfig,
    ProcessingMode, ExtractionStage
)
from ..core.exceptions import ExtractionError


@dataclass
class CognitiveStep:
    """A step in the cognitive processing pipeline"""
    name: str
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    reasoning: str
    execution_time_ms: float


class CognitivePipeline:
    """Pipeline for cognitive processing using Copilot"""
    
    def __init__(self, context: ExtractionContext):
        self.context = context
        self.steps: List[CognitiveStep] = []
        self.session_id = f"dataminer_{context.request.request_id}"
    
    async def execute_understanding_phase(self) -> CognitiveStep:
        """Use Copilot to understand the extraction task"""
        start_time = datetime.now()
        
        try:
            # Prepare understanding prompt
            content = self._prepare_content_for_analysis()
            schema_description = self._describe_target_schema()
            
            understanding_prompt = f"""
I need to extract structured data from the following content according to a specific schema.

Target Schema: {schema_description}

Content to analyze:
{content}

Please help me understand:
1. What type of content is this?
2. How well does it match the target schema?
3. What extraction strategy would work best?
4. What challenges might I face?
5. What additional context would be helpful?

Provide a comprehensive analysis of the extraction task.
"""
            
            # Use Copilot for understanding
            response = await self.context.copilot_workflow.process_message(
                understanding_prompt,
                session_id=self.session_id
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="understanding",
                description="Analyze content and extraction requirements",
                input_data={"content_length": len(content), "schema": schema_description},
                output_data=response.content,
                confidence=response.confidence,
                reasoning="Used Copilot's multi-hypothesis thinking to understand the task",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="understanding",
                description="Analyze content and extraction requirements",
                input_data=None,
                output_data=None,
                confidence=0.0,
                reasoning=f"Understanding phase failed: {str(e)}",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
    
    async def execute_planning_phase(self, understanding_result: str) -> CognitiveStep:
        """Use Copilot to plan the extraction approach"""
        start_time = datetime.now()
        
        try:
            planning_prompt = f"""
Based on the previous understanding of the content and extraction requirements:

{understanding_result}

Please create a detailed extraction plan that includes:
1. The step-by-step approach to extract the data
2. Specific techniques or patterns to look for
3. How to handle edge cases or missing information
4. Quality checks to ensure accuracy
5. Fallback strategies if the primary approach fails

Create a comprehensive extraction plan.
"""
            
            response = await self.context.copilot_workflow.process_message(
                planning_prompt,
                session_id=self.session_id
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="planning",
                description="Create detailed extraction plan",
                input_data=understanding_result,
                output_data=response.content,
                confidence=response.confidence,
                reasoning="Used Copilot's planning capabilities to create extraction strategy",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="planning",
                description="Create detailed extraction plan",
                input_data=understanding_result,
                output_data=None,
                confidence=0.0,
                reasoning=f"Planning phase failed: {str(e)}",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
    
    async def execute_extraction_phase(self, plan: str) -> CognitiveStep:
        """Execute the actual data extraction with cognitive guidance"""
        start_time = datetime.now()
        
        try:
            content = self._prepare_content_for_analysis()
            schema_name = self.context.request.schema_model.__name__
            
            extraction_prompt = f"""
Now execute the extraction plan:

Plan: {plan}

Content to extract from:
{content}

Target schema: {schema_name}

Please extract the structured data following the plan. Be thorough and accurate.
Ensure the result conforms to the target schema structure.

Important: Respond with the extracted data in the exact format specified by the schema.
"""
            
            # Use LLM with structured output for the actual extraction
            from src.llm import Message, MessageRole
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=f"You are executing a carefully planned data extraction. Follow the plan exactly and extract data according to the {schema_name} schema."
                ),
                Message(
                    role=MessageRole.USER,
                    content=extraction_prompt
                )
            ]
            
            # Get structured response
            response = await self.context.llm_client.complete(
                messages=messages,
                response_model=self.context.request.schema_model,
                temperature=0.1  # Low temperature for accuracy
            )
            
            extraction_data = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="extraction",
                description="Execute planned data extraction",
                input_data=plan,
                output_data=extraction_data,
                confidence=0.8,  # High confidence due to cognitive planning
                reasoning="Executed extraction following cognitive plan with structured output",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="extraction",
                description="Execute planned data extraction",
                input_data=plan,
                output_data=None,
                confidence=0.0,
                reasoning=f"Extraction phase failed: {str(e)}",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
    
    async def execute_validation_phase(self, extracted_data: Dict[str, Any]) -> CognitiveStep:
        """Validate and refine the extracted data with cognitive analysis"""
        start_time = datetime.now()
        
        try:
            validation_prompt = f"""
Please validate and refine this extracted data:

Extracted data:
{json.dumps(extracted_data, indent=2)}

Validation tasks:
1. Check completeness - are all important fields filled?
2. Check accuracy - does the data make sense?
3. Check consistency - are related fields consistent?
4. Identify any missing or incorrect information
5. Suggest improvements or corrections

Provide an analysis of the data quality and any recommended improvements.
"""
            
            response = await self.context.copilot_workflow.process_message(
                validation_prompt,
                session_id=self.session_id
            )
            
            # Try to improve the data based on validation feedback
            improved_data = await self._improve_data_with_feedback(
                extracted_data, 
                response.content
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="validation",
                description="Validate and refine extracted data",
                input_data=extracted_data,
                output_data=improved_data or extracted_data,
                confidence=response.confidence,
                reasoning="Used Copilot's reasoning to validate and improve data quality",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            step = CognitiveStep(
                name="validation",
                description="Validate and refine extracted data", 
                input_data=extracted_data,
                output_data=extracted_data,  # Return original data on failure
                confidence=0.5,  # Moderate confidence without validation
                reasoning=f"Validation phase failed, using original data: {str(e)}",
                execution_time_ms=execution_time
            )
            
            self.steps.append(step)
            return step
    
    def _prepare_content_for_analysis(self) -> str:
        """Prepare content for cognitive analysis"""
        if isinstance(self.context.request.content, list):
            return "\n\n---CONTENT SECTION---\n\n".join(str(item) for item in self.context.request.content)
        else:
            return str(self.context.request.content)
    
    def _describe_target_schema(self) -> str:
        """Create a description of the target schema"""
        schema_model = self.context.request.schema_model
        schema_name = schema_model.__name__
        
        # Get field information
        field_descriptions = []
        if hasattr(schema_model, '__fields__'):
            for field_name, field_info in schema_model.__fields__.items():
                description = getattr(field_info, 'description', '') or field_name
                field_type = str(field_info.annotation) if hasattr(field_info, 'annotation') else 'Any'
                field_descriptions.append(f"- {field_name} ({field_type}): {description}")
        
        field_desc_text = "\n".join(field_descriptions)
        
        return f"""
Schema: {schema_name}

Fields:
{field_desc_text}
"""
    
    async def _improve_data_with_feedback(self, data: Dict[str, Any], feedback: str) -> Optional[Dict[str, Any]]:
        """Improve data based on validation feedback"""
        try:
            improvement_prompt = f"""
Based on the validation feedback, improve this data:

Original data:
{json.dumps(data, indent=2)}

Feedback:
{feedback}

Please provide the improved data that addresses the feedback while maintaining accuracy.
"""
            
            from src.llm import Message, MessageRole
            
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="Improve the extracted data based on validation feedback. Maintain accuracy while addressing identified issues."
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
            self.context.add_warning(f"Data improvement failed: {e}")
            return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive pipeline execution"""
        return {
            "steps_executed": len(self.steps),
            "total_execution_time_ms": sum(step.execution_time_ms for step in self.steps),
            "average_confidence": sum(step.confidence for step in self.steps) / len(self.steps) if self.steps else 0.0,
            "successful_steps": sum(1 for step in self.steps if step.confidence > 0.5),
            "step_details": [
                {
                    "name": step.name,
                    "confidence": step.confidence,
                    "execution_time_ms": step.execution_time_ms,
                    "success": step.confidence > 0.5
                }
                for step in self.steps
            ]
        }


class CognitiveExtractionStrategy(BaseExtractionStrategy[T]):
    """Cognitive extraction strategy using Copilot's reasoning capabilities"""
    
    def __init__(self):
        super().__init__("CognitiveExtraction")
    
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Supports COGNITIVE and HYBRID modes"""
        return mode in [ProcessingMode.COGNITIVE, ProcessingMode.HYBRID]
    
    async def extract(
        self,
        request: ExtractionRequest[T],
        config: ExtractionConfig
    ) -> ExtractionResult[T]:
        """Execute cognitive extraction using Copilot reasoning"""
        
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
        
        if not context.copilot_workflow:
            context.add_warning("Copilot not available, falling back to standard extraction")
            return await self._fallback_extraction(context)
        
        # Create cognitive pipeline
        pipeline = CognitivePipeline(context)
        final_result = None
        
        try:
            # Understanding phase
            understanding_stage = ExtractionStage.DISCOVERY
            progress = context.get_stage_progress(understanding_stage)
            progress.start()
            await self._notify_stage_started(understanding_stage, progress)
            
            understanding_step = await pipeline.execute_understanding_phase()
            
            if understanding_step.confidence > 0.3:
                progress.complete()
                context.set_stage_confidence(understanding_stage, understanding_step.confidence)
            else:
                progress.fail("Understanding phase failed")
                context.add_error("Cognitive understanding failed")
                return await self._create_result(context, None)
            
            await self._notify_stage_completed(understanding_stage, progress)
            
            # Planning phase
            planning_stage = ExtractionStage.INITIAL  # Map to initial stage
            progress = context.get_stage_progress(planning_stage)
            progress.start()
            await self._notify_stage_started(planning_stage, progress)
            
            planning_step = await pipeline.execute_planning_phase(understanding_step.output_data)
            
            if planning_step.confidence > 0.3:
                progress.complete()
                context.set_stage_confidence(planning_stage, planning_step.confidence)
            else:
                progress.fail("Planning phase failed")
                context.add_error("Cognitive planning failed")
            
            await self._notify_stage_completed(planning_stage, progress)
            
            # Extraction phase
            extraction_stage = ExtractionStage.REFINEMENT
            progress = context.get_stage_progress(extraction_stage)
            progress.start()
            await self._notify_stage_started(extraction_stage, progress)
            
            extraction_step = await pipeline.execute_extraction_phase(planning_step.output_data)
            
            if extraction_step.confidence > 0.3 and extraction_step.output_data:
                progress.complete()
                context.set_stage_confidence(extraction_stage, extraction_step.confidence)
                context.raw_extractions.append(extraction_step.output_data)
                final_result = extraction_step.output_data
            else:
                progress.fail("Extraction phase failed")
                context.add_error("Cognitive extraction failed")
            
            await self._notify_stage_completed(extraction_stage, progress)
            
            # Validation phase (if we have results)
            if final_result:
                validation_stage = ExtractionStage.VALIDATION
                progress = context.get_stage_progress(validation_stage)
                progress.start()
                await self._notify_stage_started(validation_stage, progress)
                
                validation_step = await pipeline.execute_validation_phase(final_result)
                
                if validation_step.confidence > 0.3:
                    progress.complete()
                    context.set_stage_confidence(validation_stage, validation_step.confidence)
                    if validation_step.output_data:
                        final_result = validation_step.output_data
                    
                    # Apply cognitive confidence boost
                    for stage_key in context.confidence_scores:
                        context.confidence_scores[stage_key] += config.copilot_confidence_boost
                        context.confidence_scores[stage_key] = min(1.0, context.confidence_scores[stage_key])
                else:
                    progress.fail("Validation phase failed")
                    context.add_warning("Cognitive validation failed")
                
                await self._notify_stage_completed(validation_stage, progress)
            
            # Convert to schema instance
            final_instance = None
            if final_result:
                try:
                    final_instance = request.schema_model(**final_result)
                except Exception as e:
                    context.add_warning(f"Failed to create schema instance: {e}")
            
            # Add pipeline summary to context
            pipeline_summary = pipeline.get_pipeline_summary()
            context.sources_processed.append("cognitive_pipeline")
            
            return await self._create_result(context, final_instance)
            
        except Exception as e:
            context.add_error(f"Cognitive extraction failed: {str(e)}")
            return await self._create_result(context, None)
    
    async def _fallback_extraction(self, context: ExtractionContext) -> ExtractionResult[T]:
        """Fallback to simple extraction when Copilot is not available"""
        try:
            from .simple import SimpleExtractionStrategy
            
            simple_strategy = SimpleExtractionStrategy()
            return await simple_strategy.extract(context.request, context.config)
            
        except Exception as e:
            context.add_error(f"Fallback extraction failed: {e}")
            return await self._create_result(context, None)