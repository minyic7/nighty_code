# src/dataminer/strategies/reflective.py
"""
Reflective extraction strategy that improves iteratively
"""

from typing import Dict, List, Optional, Any, Type, TypeVar
from pydantic import BaseModel
import asyncio
import logging

from ..core.types import (
    ExtractionResult, ConfidenceMetrics, GapAnalysis,
    ProcessingMode, ExtractionStage
)
from .base import BaseExtractionStrategy, ExtractionContext

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ReflectiveExtractionStrategy(BaseExtractionStrategy):
    """
    Strategy that reflects on extraction quality and iteratively improves
    
    Process:
    1. Initial extraction attempt
    2. Analyze quality and gaps
    3. Generate improvement hypotheses
    4. Re-extract problematic areas
    5. Merge and validate improvements
    6. Repeat until confidence threshold met or max iterations reached
    """
    
    def __init__(self, max_iterations: int = 3, confidence_threshold: float = 0.8):
        super().__init__(name="reflective")
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.extraction_history: List[ExtractionResult] = []
        self.supported_modes = [ProcessingMode.THOROUGH, ProcessingMode.COGNITIVE]
    
    def supports_mode(self, mode: ProcessingMode) -> bool:
        """Check if strategy supports processing mode"""
        return mode in self.supported_modes
        
    async def extract(
        self,
        context: ExtractionContext,
        schema: Type[T]
    ) -> ExtractionResult[T]:
        """Execute reflective extraction with iterative improvement"""
        
        logger.info(f"Starting reflective extraction for {schema.__name__}")
        
        # Track iterations
        iteration = 0
        best_result = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Perform extraction
            if iteration == 1:
                # First iteration: standard extraction
                result = await self._initial_extraction(context, schema)
            else:
                # Subsequent iterations: targeted improvement
                result = await self._improved_extraction(
                    context, schema, best_result
                )
            
            # Store in history
            self.extraction_history.append(result)
            
            # Update best result if improved
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result
            
            # Check if we've reached confidence threshold
            if result.confidence >= self.confidence_threshold:
                logger.info(f"Reached confidence threshold: {result.confidence:.2f}")
                break
            
            # Analyze what needs improvement
            gaps = await self._analyze_gaps(result, schema)
            
            if not gaps.missing_fields and not gaps.low_confidence_fields:
                logger.info("No significant gaps found, stopping iteration")
                break
            
            logger.info(f"Gaps found: {len(gaps.missing_fields)} missing, "
                       f"{len(gaps.low_confidence_fields)} low confidence")
        
        # Final reflection and summary
        best_result = await self._final_reflection(best_result, schema)
        
        return best_result
    
    async def _initial_extraction(
        self,
        context: ExtractionContext,
        schema: Type[T]
    ) -> ExtractionResult[T]:
        """Perform initial extraction attempt"""
        
        # Use LLM for extraction
        from src.llm import Message, MessageRole
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "Extract structured data from code. Be thorough and precise."
            ),
            Message(
                MessageRole.USER,
                f"Extract {schema.__name__} from:\n\n{context.content[:3000]}"
            )
        ]
        
        try:
            # Extract with Instructor
            data = await context.llm_client.complete(
                messages=messages,
                response_model=schema,
                temperature=0.3
            )
            
            # Calculate initial confidence
            confidence = ConfidenceMetrics(
                extraction_quality=0.7,
                schema_compliance=0.8,
                completeness=0.6,
                consistency=0.7
            )
            confidence.update_overall()
            
            return ExtractionResult(
                success=True,
                data=data,
                confidence=confidence.overall,
                confidence_metrics=confidence,
                metadata={"iteration": 1, "stage": "initial"}
            )
            
        except Exception as e:
            logger.error(f"Initial extraction failed: {e}")
            return ExtractionResult(
                success=False,
                error=str(e),
                confidence=0.0,
                metadata={"iteration": 1, "stage": "initial", "error": str(e)}
            )
    
    async def _improved_extraction(
        self,
        context: ExtractionContext,
        schema: Type[T],
        previous_result: ExtractionResult[T]
    ) -> ExtractionResult[T]:
        """Perform improved extraction based on previous results"""
        
        # Analyze what went wrong
        gaps = previous_result.gaps or []
        low_conf_fields = previous_result.field_confidence or {}
        
        # Generate improvement prompt
        improvement_prompt = self._generate_improvement_prompt(
            schema, gaps, low_conf_fields, previous_result
        )
        
        from src.llm import Message, MessageRole
        
        messages = [
            Message(
                MessageRole.SYSTEM,
                "You are improving a previous extraction. Focus on the gaps and low-confidence areas."
            ),
            Message(
                MessageRole.USER,
                improvement_prompt
            )
        ]
        
        try:
            # Re-extract with focus on improvements
            improved_data = await context.llm_client.complete(
                messages=messages,
                response_model=schema,
                temperature=0.4  # Slightly higher for creativity
            )
            
            # Merge with previous results
            merged_data = await self._merge_results(
                previous_result.data, improved_data, low_conf_fields
            )
            
            # Recalculate confidence
            confidence = await self._calculate_improved_confidence(
                merged_data, previous_result.confidence_metrics
            )
            
            return ExtractionResult(
                success=True,
                data=merged_data,
                confidence=confidence.overall,
                confidence_metrics=confidence,
                metadata={
                    "iteration": len(self.extraction_history) + 1,
                    "stage": "improved",
                    "improvements": len(gaps)
                }
            )
            
        except Exception as e:
            logger.error(f"Improved extraction failed: {e}")
            # Return previous result if improvement fails
            return previous_result
    
    async def _analyze_gaps(
        self,
        result: ExtractionResult[T],
        schema: Type[T]
    ) -> GapAnalysis:
        """Analyze gaps in extraction"""
        
        gaps = GapAnalysis()
        
        if not result.data:
            gaps.missing_fields = list(schema.model_fields.keys())
            return gaps
        
        # Check for None/empty values
        data_dict = result.data.model_dump()
        
        for field_name, field_info in schema.model_fields.items():
            value = data_dict.get(field_name)
            
            # Check if required field is missing
            if field_info.is_required() and (value is None or value == [] or value == {}):
                gaps.missing_fields.append(field_name)
            
            # Check confidence if available
            if result.field_confidence:
                field_conf = result.field_confidence.get(field_name, 1.0)
                if field_conf < 0.6:
                    gaps.low_confidence_fields.append(field_name)
        
        # Generate recommendations
        if gaps.missing_fields:
            gaps.recommended_actions.append(
                f"Search for patterns related to: {', '.join(gaps.missing_fields[:3])}"
            )
        
        if gaps.low_confidence_fields:
            gaps.recommended_actions.append(
                f"Verify and improve: {', '.join(gaps.low_confidence_fields[:3])}"
            )
        
        return gaps
    
    def _generate_improvement_prompt(
        self,
        schema: Type[T],
        gaps: List[str],
        low_conf_fields: Dict[str, float],
        previous_result: ExtractionResult[T]
    ) -> str:
        """Generate a prompt for improvement based on gaps"""
        
        prompt = f"Improve extraction of {schema.__name__}.\n\n"
        
        if previous_result.data:
            prompt += "Previous extraction (partial):\n"
            prompt += f"{previous_result.data.model_dump_json(indent=2)[:1000]}\n\n"
        
        prompt += "Focus on improving:\n"
        
        if gaps:
            prompt += f"- Missing fields: {', '.join(gaps[:5])}\n"
        
        if low_conf_fields:
            low_fields = [f"{k} (conf: {v:.1%})" 
                         for k, v in low_conf_fields.items() if v < 0.6][:5]
            prompt += f"- Low confidence fields: {', '.join(low_fields)}\n"
        
        prompt += "\nLook for:\n"
        prompt += "- Function signatures with parameters\n"
        prompt += "- Nested structures and relationships\n"
        prompt += "- Configuration values and settings\n"
        prompt += "- Edge cases and special patterns\n"
        
        prompt += f"\nContent to analyze:\n{previous_result.metadata.get('content', '')[:2000]}"
        
        return prompt
    
    async def _merge_results(
        self,
        original: Optional[T],
        improved: T,
        low_conf_fields: Dict[str, float]
    ) -> T:
        """Merge improved results with original, keeping best values"""
        
        if not original:
            return improved
        
        # Convert to dicts for merging
        orig_dict = original.model_dump()
        imp_dict = improved.model_dump()
        
        merged = {}
        
        for field in orig_dict.keys():
            orig_val = orig_dict.get(field)
            imp_val = imp_dict.get(field)
            
            # Decide which value to keep
            if field in low_conf_fields and low_conf_fields[field] < 0.6:
                # Low confidence field - prefer improved
                merged[field] = imp_val if imp_val else orig_val
            elif orig_val and not imp_val:
                # Original has value, improved doesn't
                merged[field] = orig_val
            elif imp_val and not orig_val:
                # Improved has value, original doesn't
                merged[field] = imp_val
            elif isinstance(orig_val, list) and isinstance(imp_val, list):
                # Merge lists, removing duplicates
                combined = orig_val + imp_val
                # Simple dedup (works for primitive types)
                merged[field] = list(dict.fromkeys(combined))
            elif isinstance(orig_val, dict) and isinstance(imp_val, dict):
                # Merge dicts
                merged[field] = {**orig_val, **imp_val}
            else:
                # Default to improved value
                merged[field] = imp_val if imp_val else orig_val
        
        # Create new instance with merged data
        return type(original)(**merged)
    
    async def _calculate_improved_confidence(
        self,
        data: T,
        previous_metrics: Optional[ConfidenceMetrics]
    ) -> ConfidenceMetrics:
        """Calculate confidence for improved extraction"""
        
        metrics = ConfidenceMetrics()
        
        if not data:
            return metrics
        
        # Count filled fields
        data_dict = data.model_dump()
        total_fields = len(data_dict)
        filled_fields = sum(1 for v in data_dict.values() 
                          if v is not None and v != [] and v != {})
        
        # Calculate completeness
        metrics.completeness = filled_fields / total_fields if total_fields > 0 else 0
        
        # Improvement bonus
        if previous_metrics:
            improvement = metrics.completeness - previous_metrics.completeness
            if improvement > 0:
                metrics.extraction_quality = min(0.9, previous_metrics.extraction_quality + improvement)
            else:
                metrics.extraction_quality = previous_metrics.extraction_quality
        else:
            metrics.extraction_quality = 0.7
        
        # Set other metrics
        metrics.schema_compliance = 0.85  # Instructor ensures this
        metrics.consistency = 0.8
        
        metrics.update_overall()
        return metrics
    
    async def _final_reflection(
        self,
        result: ExtractionResult[T],
        schema: Type[T]
    ) -> ExtractionResult[T]:
        """Final reflection and quality assessment"""
        
        # Add extraction history summary
        result.metadata["total_iterations"] = len(self.extraction_history)
        result.metadata["confidence_progression"] = [
            r.confidence for r in self.extraction_history
        ]
        
        # Final gap analysis
        final_gaps = await self._analyze_gaps(result, schema)
        result.gaps = final_gaps.missing_fields
        result.recommendations = final_gaps.recommended_actions
        
        # Add reflection notes
        if result.confidence < 0.5:
            result.recommendations.append(
                "Low confidence extraction. Consider: "
                "1) Providing more context, "
                "2) Simplifying schema, "
                "3) Using different extraction strategy"
            )
        elif result.confidence < 0.7:
            result.recommendations.append(
                "Moderate confidence. Review extracted data and verify critical fields."
            )
        else:
            result.recommendations.append(
                "High confidence extraction. Spot-check for accuracy."
            )
        
        logger.info(f"Final extraction confidence: {result.confidence:.2f}")
        logger.info(f"Iterations used: {len(self.extraction_history)}")
        
        return result