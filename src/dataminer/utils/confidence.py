# src/dataminer/utils/confidence.py
"""Confidence scoring and quality assessment utilities"""

from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
import re
import statistics
from collections import Counter

from ..models.base import ExtractionSchema
from ..core.types import ConfidenceMetrics, GapAnalysis, ConfidenceLevel


@dataclass
class QualityMetrics:
    """Detailed quality metrics for extracted data"""
    
    # Content quality
    completeness_score: float = 0.0          # How complete is the data
    accuracy_score: float = 0.0              # How accurate is the data
    consistency_score: float = 0.0           # How consistent is the data
    relevance_score: float = 0.0             # How relevant is the extracted data
    
    # Technical quality
    schema_compliance: float = 0.0           # How well does data match schema
    type_correctness: float = 0.0            # Type validation score
    format_compliance: float = 0.0           # Format compliance score
    
    # Field-level analysis
    field_scores: Dict[str, float] = field(default_factory=dict)
    empty_fields: List[str] = field(default_factory=list)
    low_quality_fields: List[str] = field(default_factory=list)
    
    # Content analysis
    text_quality_score: float = 0.0         # Quality of text content
    data_density_score: float = 0.0         # Information density
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score"""
        components = [
            self.completeness_score * 0.25,
            self.accuracy_score * 0.25,
            self.consistency_score * 0.20,
            self.schema_compliance * 0.15,
            self.relevance_score * 0.15
        ]
        return sum(components)


class ConfidenceCalculator:
    """Calculate confidence scores for extracted data"""
    
    def __init__(self):
        self.text_quality_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            'version': r'\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?',
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}',
            'phone': r'(\+\d{1,3})?[\s-.]?\(?\d{3}\)?[\s-.]?\d{3}[\s-.]?\d{4}',
            'uuid': r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
        }
    
    async def calculate_confidence(
        self,
        extracted_data: ExtractionSchema,
        source_content: Union[str, List[str]],
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics"""
        
        # Initialize confidence metrics
        metrics = ConfidenceMetrics()
        
        # Calculate quality metrics first
        quality_metrics = await self.calculate_quality_metrics(extracted_data, source_content)
        
        # Map quality metrics to confidence components
        metrics.extraction_quality = quality_metrics.calculate_overall_quality()
        metrics.schema_compliance = quality_metrics.schema_compliance
        metrics.completeness = quality_metrics.completeness_score
        metrics.consistency = quality_metrics.consistency_score
        
        # Calculate field-level confidence
        metrics.field_confidence = quality_metrics.field_scores.copy()
        
        # Calculate validation scores
        validation_result = extracted_data.validate_completeness()
        metrics.validation_scores = {
            'completeness_validation': validation_result.get('completeness_score', 0.0),
            'required_fields_present': 1.0 - (len(validation_result.get('missing_required', [])) / 
                                             max(1, validation_result.get('total_required_fields', 1)))
        }
        
        # Update overall confidence
        metrics.update_overall()
        
        return metrics
    
    async def calculate_quality_metrics(
        self,
        extracted_data: ExtractionSchema,
        source_content: Union[str, List[str]]
    ) -> QualityMetrics:
        """Calculate detailed quality metrics"""
        
        metrics = QualityMetrics()
        
        # Get data as dictionary
        if hasattr(extracted_data, 'model_dump'):
            data_dict = extracted_data.model_dump()
        else:
            data_dict = extracted_data.__dict__
        
        # Calculate completeness
        metrics.completeness_score = await self._calculate_completeness(extracted_data, data_dict)
        
        # Calculate schema compliance
        metrics.schema_compliance = await self._calculate_schema_compliance(extracted_data, data_dict)
        
        # Calculate consistency
        metrics.consistency_score = await self._calculate_consistency(data_dict)
        
        # Calculate relevance
        metrics.relevance_score = await self._calculate_relevance(data_dict, source_content)
        
        # Calculate text quality
        metrics.text_quality_score = await self._calculate_text_quality(data_dict)
        
        # Calculate field-level scores
        metrics.field_scores = await self._calculate_field_scores(data_dict, source_content)
        
        # Identify problematic fields
        metrics.empty_fields = [k for k, v in data_dict.items() if self._is_empty_value(v)]
        metrics.low_quality_fields = [k for k, score in metrics.field_scores.items() if score < 0.5]
        
        return metrics
    
    async def _calculate_completeness(self, schema_instance: ExtractionSchema, data_dict: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        try:
            validation_result = schema_instance.validate_completeness()
            return validation_result.get('completeness_score', 0.0)
        except Exception:
            # Fallback calculation
            total_fields = len(data_dict)
            empty_fields = sum(1 for v in data_dict.values() if self._is_empty_value(v))
            return (total_fields - empty_fields) / total_fields if total_fields > 0 else 0.0
    
    async def _calculate_schema_compliance(self, schema_instance: ExtractionSchema, data_dict: Dict[str, Any]) -> float:
        """Calculate schema compliance score"""
        try:
            # Try to recreate instance to validate compliance
            schema_type = type(schema_instance)
            test_instance = schema_type(**data_dict)
            return 1.0  # If successful, full compliance
        except Exception as e:
            # Calculate partial compliance based on field validation
            schema_fields = getattr(schema_type, '__fields__', {})
            valid_fields = 0
            total_fields = len(schema_fields)
            
            for field_name, field_info in schema_fields.items():
                value = data_dict.get(field_name)
                if self._validate_field_type(value, field_info):
                    valid_fields += 1
            
            return valid_fields / total_fields if total_fields > 0 else 0.0
    
    async def _calculate_consistency(self, data_dict: Dict[str, Any]) -> float:
        """Calculate internal consistency score"""
        consistency_checks = []
        
        # Check for contradictory information
        # This is a basic implementation - could be enhanced with domain-specific rules
        
        # Check date consistency
        date_fields = [k for k, v in data_dict.items() if 'date' in k.lower() and isinstance(v, str)]
        if len(date_fields) >= 2:
            dates = []
            for field in date_fields:
                date_str = data_dict[field]
                parsed_date = self._parse_date(date_str)
                if parsed_date:
                    dates.append(parsed_date)
            
            if len(dates) >= 2:
                # Check if dates are in reasonable order
                sorted_dates = sorted(dates)
                is_consistent = dates == sorted_dates or dates == sorted_dates[::-1]
                consistency_checks.append(1.0 if is_consistent else 0.5)
        
        # Check numerical consistency
        numerical_fields = [k for k, v in data_dict.items() if isinstance(v, (int, float))]
        if len(numerical_fields) >= 2:
            # Basic sanity checks for numerical values
            values = [data_dict[field] for field in numerical_fields]
            # Check for reasonable ranges (no extreme outliers)
            if len(values) >= 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                outliers = sum(1 for v in values if abs(v - mean_val) > 3 * std_val)
                consistency_checks.append(1.0 - (outliers / len(values)))
        
        # Check string formatting consistency
        string_fields = [k for k, v in data_dict.items() if isinstance(v, str) and v.strip()]
        if len(string_fields) >= 2:
            # Check for consistent formatting patterns
            case_patterns = [self._detect_case_pattern(data_dict[field]) for field in string_fields]
            case_consistency = len(set(case_patterns)) / len(case_patterns)
            consistency_checks.append(1.0 - case_consistency + 0.5)  # Partial credit for variety
        
        return statistics.mean(consistency_checks) if consistency_checks else 0.8
    
    async def _calculate_relevance(self, data_dict: Dict[str, Any], source_content: Union[str, List[str]]) -> float:
        """Calculate relevance of extracted data to source content"""
        if isinstance(source_content, list):
            combined_content = " ".join(str(item) for item in source_content)
        else:
            combined_content = str(source_content)
        
        combined_content = combined_content.lower()
        relevance_scores = []
        
        for field_name, value in data_dict.items():
            if self._is_empty_value(value):
                continue
            
            value_str = str(value).lower()
            
            # Direct match
            if value_str in combined_content:
                relevance_scores.append(1.0)
            else:
                # Partial match based on words
                value_words = set(re.findall(r'\b\w+\b', value_str))
                content_words = set(re.findall(r'\b\w+\b', combined_content))
                
                if value_words and content_words:
                    overlap = len(value_words.intersection(content_words))
                    relevance_scores.append(overlap / len(value_words))
                else:
                    relevance_scores.append(0.0)
        
        return statistics.mean(relevance_scores) if relevance_scores else 0.0
    
    async def _calculate_text_quality(self, data_dict: Dict[str, Any]) -> float:
        """Calculate quality of text content"""
        text_fields = [v for v in data_dict.values() if isinstance(v, str) and len(v.strip()) > 0]
        
        if not text_fields:
            return 0.0
        
        quality_scores = []
        
        for text in text_fields:
            score = 0.0
            
            # Length check (not too short, not too long for the type)
            if 5 <= len(text) <= 1000:
                score += 0.2
            elif len(text) > 1000:
                score += 0.1
            
            # Grammar and structure (basic checks)
            # Proper sentence structure
            sentences = re.split(r'[.!?]+', text)
            if len(sentences) > 1:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if 5 <= avg_sentence_length <= 30:
                    score += 0.2
            
            # Capitalization
            if text[0].isupper():
                score += 0.1
            
            # No excessive repetition
            words = text.lower().split()
            if len(words) > 0:
                word_counts = Counter(words)
                max_repetition = max(word_counts.values()) if word_counts else 0
                if max_repetition / len(words) < 0.3:  # No word appears more than 30% of the time
                    score += 0.2
            
            # Presence of meaningful content (contains letters and possibly numbers)
            if re.search(r'[a-zA-Z]', text):
                score += 0.1
            
            # Format-specific quality checks
            for pattern_name, pattern in self.text_quality_patterns.items():
                if re.search(pattern, text):
                    score += 0.1
                    break
            
            quality_scores.append(min(score, 1.0))
        
        return statistics.mean(quality_scores)
    
    async def _calculate_field_scores(self, data_dict: Dict[str, Any], source_content: Union[str, List[str]]) -> Dict[str, float]:
        """Calculate confidence score for each field"""
        field_scores = {}
        
        if isinstance(source_content, list):
            combined_content = " ".join(str(item) for item in source_content)
        else:
            combined_content = str(source_content)
        
        for field_name, value in data_dict.items():
            score = 0.0
            
            # Empty value check
            if self._is_empty_value(value):
                field_scores[field_name] = 0.0
                continue
            
            # Type appropriateness
            if isinstance(value, str):
                if len(value.strip()) > 0:
                    score += 0.3
                
                # Check if value appears in source
                if value.lower() in combined_content.lower():
                    score += 0.4
                
                # Check for format patterns
                for pattern in self.text_quality_patterns.values():
                    if re.search(pattern, value):
                        score += 0.2
                        break
            
            elif isinstance(value, (int, float)):
                score += 0.3  # Numbers are generally reliable if extracted
                
                # Check if number appears in source
                if str(value) in combined_content:
                    score += 0.4
            
            elif isinstance(value, list):
                if len(value) > 0:
                    score += 0.3
                    
                    # Check if list items appear in source
                    matches = sum(1 for item in value if str(item).lower() in combined_content.lower())
                    if matches > 0:
                        score += 0.4 * (matches / len(value))
            
            elif isinstance(value, dict):
                if len(value) > 0:
                    score += 0.3
                    
                    # Check if dict values appear in source
                    dict_values = [str(v) for v in value.values()]
                    matches = sum(1 for v in dict_values if v.lower() in combined_content.lower())
                    if matches > 0:
                        score += 0.4 * (matches / len(dict_values))
            
            else:
                score += 0.2  # Default score for other types
            
            field_scores[field_name] = min(score, 1.0)
        
        return field_scores
    
    def _is_empty_value(self, value: Any) -> bool:
        """Check if a value is considered empty"""
        if value is None:
            return True
        if isinstance(value, str):
            return len(value.strip()) == 0
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False
    
    def _validate_field_type(self, value: Any, field_info: Any) -> bool:
        """Validate if value matches expected field type"""
        # Basic type validation - could be enhanced
        if hasattr(field_info, 'annotation'):
            expected_type = field_info.annotation
            
            # Handle Optional types
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                # Check if value matches any of the union types
                return any(isinstance(value, arg) for arg in expected_type.__args__ if arg != type(None))
            
            return isinstance(value, expected_type)
        
        return True  # If no type info, assume valid
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object"""
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _detect_case_pattern(self, text: str) -> str:
        """Detect the case pattern of text"""
        if text.isupper():
            return 'upper'
        elif text.islower():
            return 'lower'
        elif text.istitle():
            return 'title'
        elif text[0].isupper() and text[1:].islower():
            return 'sentence'
        else:
            return 'mixed'
    
    async def perform_gap_analysis(
        self,
        extracted_data: ExtractionSchema,
        confidence_metrics: ConfidenceMetrics,
        context: Optional[Dict[str, Any]] = None
    ) -> GapAnalysis:
        """Perform comprehensive gap analysis"""
        
        gap_analysis = GapAnalysis()
        
        # Get validation results
        validation_result = extracted_data.validate_completeness()
        
        # Missing fields
        gap_analysis.missing_fields = validation_result.get('missing_required', [])
        
        # Incomplete fields (fields with low confidence)
        gap_analysis.incomplete_fields = [
            field for field, confidence in confidence_metrics.field_confidence.items()
            if confidence < 0.5
        ]
        
        # Low confidence fields
        gap_analysis.low_confidence_fields = [
            field for field, confidence in confidence_metrics.field_confidence.items()
            if 0.3 <= confidence < 0.7
        ]
        
        # Calculate completeness metrics
        total_fields = len(extracted_data.get_required_fields())
        missing_count = len(gap_analysis.missing_fields)
        gap_analysis.completeness_score = (total_fields - missing_count) / total_fields if total_fields else 1.0
        gap_analysis.coverage_percentage = gap_analysis.completeness_score * 100
        
        # Generate recommendations
        if gap_analysis.missing_fields:
            gap_analysis.recommended_actions.extend([
                "Provide more comprehensive input data",
                "Check if missing information exists in related sources",
                "Consider using multi-stage extraction for complex schemas"
            ])
        
        if gap_analysis.incomplete_fields:
            gap_analysis.recommended_actions.extend([
                "Improve extraction prompts with more specific instructions",
                "Use cognitive extraction mode for better reasoning",
                "Validate source content contains the required information"
            ])
        
        if gap_analysis.low_confidence_fields:
            gap_analysis.recommended_actions.extend([
                "Review extraction results for accuracy",
                "Consider additional context or examples",
                "Use higher-quality source material"
            ])
        
        # Additional source suggestions
        if context:
            source_count = context.get('sources_processed', 0)
            if source_count < 3:
                gap_analysis.additional_sources.append("Process additional related files or documents")
            
            if context.get('repository_path') and not context.get('has_readme', False):
                gap_analysis.additional_sources.append("Include project README for additional context")
        
        return gap_analysis