# src/dataminer/utils/validator.py
"""Schema validation utilities"""

from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import inspect
from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..models.base import ExtractionSchema
from ..core.exceptions import ValidationError, SchemaError


@dataclass
class ValidationReport:
    """Comprehensive validation report for a schema"""
    
    schema_name: str
    is_valid: bool
    
    # Schema structure validation
    has_required_methods: bool = True
    has_proper_inheritance: bool = True
    field_validations: Dict[str, bool] = field(default_factory=dict)
    
    # Content validation  
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    
    # Metrics
    complexity_score: float = 0.0
    usability_score: float = 0.0
    completeness_score: float = 0.0
    
    # Metadata
    total_fields: int = 0
    required_fields: int = 0
    optional_fields: int = 0
    validated_at: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str):
        """Add validation error"""
        self.validation_errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.validation_warnings.append(warning)
    
    def add_suggestion(self, suggestion: str):
        """Add improvement suggestion"""
        self.improvement_suggestions.append(suggestion)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall schema quality score"""
        components = []
        
        if self.is_valid:
            components.append(0.4)  # Base score for valid schema
        
        # Deduct for errors and warnings
        error_penalty = min(0.3, len(self.validation_errors) * 0.1)
        warning_penalty = min(0.1, len(self.validation_warnings) * 0.02)
        
        structure_score = 0.3 if (self.has_required_methods and self.has_proper_inheritance) else 0.1
        components.append(structure_score - error_penalty - warning_penalty)
        
        # Add component scores
        components.extend([
            self.complexity_score * 0.1,
            self.usability_score * 0.1, 
            self.completeness_score * 0.1
        ])
        
        return max(0.0, sum(components))


class SchemaValidator:
    """Validates extraction schemas for quality and usability"""
    
    def __init__(self):
        self.validation_cache: Dict[str, ValidationReport] = {}
    
    async def validate_schema(self, schema_class: Type[ExtractionSchema]) -> ValidationReport:
        """Comprehensive schema validation"""
        
        schema_name = schema_class.__name__
        
        # Check cache first
        cache_key = f"{schema_name}_{hash(str(schema_class))}"
        if cache_key in self.validation_cache:
            cached_report = self.validation_cache[cache_key]
            if (datetime.now() - cached_report.validated_at).seconds < 3600:  # 1 hour cache
                return cached_report
        
        # Create validation report
        report = ValidationReport(
            schema_name=schema_name,
            is_valid=True
        )
        
        try:
            # Validate schema structure
            await self._validate_schema_structure(schema_class, report)
            
            # Validate field definitions
            await self._validate_field_definitions(schema_class, report)
            
            # Validate methods and inheritance
            await self._validate_methods_and_inheritance(schema_class, report)
            
            # Calculate complexity and usability
            await self._calculate_schema_metrics(schema_class, report)
            
            # Generate recommendations
            await self._generate_recommendations(schema_class, report)
            
            # Cache the report
            self.validation_cache[cache_key] = report
            
        except Exception as e:
            report.add_error(f"Schema validation failed: {str(e)}")
        
        return report
    
    async def validate_instance(self, instance: ExtractionSchema) -> ValidationReport:
        """Validate a specific instance of extracted data"""
        
        schema_class = type(instance)
        report = ValidationReport(
            schema_name=schema_class.__name__,
            is_valid=True
        )
        
        try:
            # Validate instance data
            await self._validate_instance_data(instance, report)
            
            # Validate completeness
            await self._validate_instance_completeness(instance, report)
            
            # Validate data quality
            await self._validate_data_quality(instance, report)
            
        except Exception as e:
            report.add_error(f"Instance validation failed: {str(e)}")
        
        return report
    
    async def compare_schemas(
        self,
        schema1: Type[ExtractionSchema], 
        schema2: Type[ExtractionSchema]
    ) -> Dict[str, Any]:
        """Compare two schemas for compatibility and differences"""
        
        comparison = {
            "schema1_name": schema1.__name__,
            "schema2_name": schema2.__name__,
            "are_compatible": True,
            "differences": [],
            "common_fields": [],
            "unique_to_schema1": [],
            "unique_to_schema2": [],
            "type_conflicts": [],
            "compatibility_score": 0.0
        }
        
        try:
            # Get field information
            fields1 = self._get_schema_fields(schema1)
            fields2 = self._get_schema_fields(schema2)
            
            # Find common and unique fields
            common_field_names = set(fields1.keys()) & set(fields2.keys())
            unique_to_1 = set(fields1.keys()) - set(fields2.keys())
            unique_to_2 = set(fields2.keys()) - set(fields1.keys())
            
            comparison["common_fields"] = list(common_field_names)
            comparison["unique_to_schema1"] = list(unique_to_1)
            comparison["unique_to_schema2"] = list(unique_to_2)
            
            # Check type conflicts
            for field_name in common_field_names:
                field1_info = fields1[field_name]
                field2_info = fields2[field_name]
                
                type1 = getattr(field1_info, 'annotation', None)
                type2 = getattr(field2_info, 'annotation', None)
                
                if type1 != type2:
                    comparison["type_conflicts"].append({
                        "field": field_name,
                        "type1": str(type1),
                        "type2": str(type2)
                    })
                    comparison["are_compatible"] = False
            
            # Calculate compatibility score
            total_fields = len(set(fields1.keys()) | set(fields2.keys()))
            common_fields = len(common_field_names)
            conflicts = len(comparison["type_conflicts"])
            
            if total_fields > 0:
                base_compatibility = common_fields / total_fields
                conflict_penalty = conflicts * 0.1
                comparison["compatibility_score"] = max(0.0, base_compatibility - conflict_penalty)
            
        except Exception as e:
            comparison["error"] = str(e)
        
        return comparison
    
    async def _validate_schema_structure(self, schema_class: Type[ExtractionSchema], report: ValidationReport):
        """Validate basic schema structure"""
        
        # Check inheritance
        if not issubclass(schema_class, ExtractionSchema):
            report.add_error("Schema must inherit from ExtractionSchema")
            report.has_proper_inheritance = False
        
        # Check if it's a Pydantic model
        if not issubclass(schema_class, BaseModel):
            report.add_error("Schema must be a Pydantic BaseModel")
            report.has_proper_inheritance = False
        
        # Check for required methods
        required_methods = ['get_confidence_fields', 'get_required_fields']
        for method_name in required_methods:
            if not hasattr(schema_class, method_name):
                report.add_error(f"Schema missing required method: {method_name}")
                report.has_required_methods = False
            elif not callable(getattr(schema_class, method_name)):
                report.add_error(f"Schema method {method_name} is not callable")
                report.has_required_methods = False
        
        # Check for class-level configuration
        if hasattr(schema_class, 'Config'):
            config = schema_class.Config
            if not hasattr(config, 'validate_assignment'):
                report.add_warning("Consider enabling validate_assignment in Config for better validation")
        else:
            report.add_suggestion("Add a Config class with validation settings")
    
    async def _validate_field_definitions(self, schema_class: Type[ExtractionSchema], report: ValidationReport):
        """Validate field definitions"""
        
        fields = self._get_schema_fields(schema_class)
        report.total_fields = len(fields)
        
        required_fields = 0
        optional_fields = 0
        
        for field_name, field_info in fields.items():
            is_valid = True
            
            # Check if field has description
            description = getattr(field_info, 'description', None)
            if not description:
                report.add_warning(f"Field '{field_name}' lacks description")
                is_valid = False
            
            # Check field type annotation
            if not hasattr(field_info, 'annotation'):
                report.add_error(f"Field '{field_name}' lacks type annotation")
                is_valid = False
            
            # Check if field is required
            is_required = getattr(field_info, 'is_required', lambda: False)()
            if callable(is_required):
                is_required = is_required()
            
            if is_required:
                required_fields += 1
            else:
                optional_fields += 1
            
            # Validate field type
            field_type = getattr(field_info, 'annotation', None)
            if field_type:
                type_validation = self._validate_field_type(field_name, field_type)
                if not type_validation['valid']:
                    report.add_warning(f"Field '{field_name}': {type_validation['message']}")
                    is_valid = False
            
            report.field_validations[field_name] = is_valid
        
        report.required_fields = required_fields
        report.optional_fields = optional_fields
        
        # Validate field balance
        if required_fields == 0:
            report.add_warning("Schema has no required fields - consider making key fields required")
        elif required_fields > len(fields) * 0.8:
            report.add_warning("Schema has too many required fields - consider making some optional")
    
    async def _validate_methods_and_inheritance(self, schema_class: Type[ExtractionSchema], report: ValidationReport):
        """Validate methods and inheritance chain"""
        
        # Test method functionality
        try:
            # Create a temporary instance to test methods
            temp_instance = schema_class()
            
            # Test get_required_fields
            required_fields = temp_instance.get_required_fields()
            if not isinstance(required_fields, list):
                report.add_error("get_required_fields must return a list")
            elif len(required_fields) == 0:
                report.add_warning("get_required_fields returns empty list")
            
            # Test get_confidence_fields  
            confidence_fields = temp_instance.get_confidence_fields()
            if not isinstance(confidence_fields, list):
                report.add_error("get_confidence_fields must return a list")
            
            # Test validate_completeness
            if hasattr(temp_instance, 'validate_completeness'):
                completeness_result = temp_instance.validate_completeness()
                if not isinstance(completeness_result, dict):
                    report.add_error("validate_completeness must return a dictionary")
                else:
                    expected_keys = ['completeness_score', 'missing_required']
                    for key in expected_keys:
                        if key not in completeness_result:
                            report.add_warning(f"validate_completeness should include '{key}' in result")
            
        except Exception as e:
            report.add_error(f"Method testing failed: {str(e)}")
    
    async def _calculate_schema_metrics(self, schema_class: Type[ExtractionSchema], report: ValidationReport):
        """Calculate schema quality metrics"""
        
        fields = self._get_schema_fields(schema_class)
        
        # Complexity score (based on field count and types)
        field_count = len(fields)
        complex_types = sum(1 for field_info in fields.values() 
                          if self._is_complex_type(getattr(field_info, 'annotation', None)))
        
        if field_count > 0:
            complexity_ratio = complex_types / field_count
            # Normalize complexity score (sweet spot around 20-30% complex fields)
            if 0.2 <= complexity_ratio <= 0.4:
                report.complexity_score = 1.0
            else:
                report.complexity_score = max(0.0, 1.0 - abs(complexity_ratio - 0.3) * 2)
        
        # Usability score (based on descriptions, naming, etc.)
        usability_factors = []
        
        # Field naming consistency
        field_names = list(fields.keys())
        snake_case_count = sum(1 for name in field_names if '_' in name and name.islower())
        naming_consistency = snake_case_count / len(field_names) if field_names else 0
        usability_factors.append(naming_consistency)
        
        # Description coverage
        described_fields = sum(1 for field_info in fields.values() 
                             if getattr(field_info, 'description', None))
        description_coverage = described_fields / len(fields) if fields else 0
        usability_factors.append(description_coverage)
        
        # Reasonable field count (not too many, not too few)
        if 3 <= field_count <= 20:
            usability_factors.append(1.0)
        elif field_count < 3:
            usability_factors.append(0.5)
        else:
            usability_factors.append(max(0.3, 1.0 - (field_count - 20) * 0.05))
        
        report.usability_score = sum(usability_factors) / len(usability_factors) if usability_factors else 0
        
        # Completeness score (has all recommended components)
        completeness_factors = []
        completeness_factors.append(1.0 if report.has_required_methods else 0.0)
        completeness_factors.append(1.0 if report.has_proper_inheritance else 0.0)
        completeness_factors.append(description_coverage)
        
        report.completeness_score = sum(completeness_factors) / len(completeness_factors)
    
    async def _generate_recommendations(self, schema_class: Type[ExtractionSchema], report: ValidationReport):
        """Generate improvement recommendations"""
        
        # Based on validation results
        if not report.has_required_methods:
            report.add_suggestion("Implement all required methods: get_confidence_fields, get_required_fields")
        
        if report.total_fields > 25:
            report.add_suggestion("Consider breaking large schema into smaller, focused schemas")
        
        if report.required_fields == 0:
            report.add_suggestion("Add required fields for essential data validation")
        
        if report.complexity_score < 0.5:
            if report.complexity_score < 0.3:
                report.add_suggestion("Schema may be too simple - consider adding structured fields")
            else:
                report.add_suggestion("Schema may be too complex - consider simplifying field types")
        
        if report.usability_score < 0.7:
            report.add_suggestion("Improve field naming consistency and add more descriptions")
        
        # Best practices
        report.best_practices = [
            "Use clear, descriptive field names in snake_case",
            "Provide meaningful descriptions for all fields",
            "Balance required vs optional fields appropriately",
            "Use appropriate field types (avoid 'Any' when possible)",
            "Include examples in field descriptions for complex types",
            "Consider field validation rules for data quality"
        ]
    
    async def _validate_instance_data(self, instance: ExtractionSchema, report: ValidationReport):
        """Validate specific instance data"""
        
        try:
            # Try to re-validate the instance
            if hasattr(instance, 'model_validate'):
                data = instance.model_dump()
                validated = type(instance).model_validate(data)
            else:
                # Fallback validation
                validated = instance
                
        except PydanticValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = error['msg']
                report.add_error(f"Field '{field}': {message}")
        except Exception as e:
            report.add_error(f"Instance validation failed: {str(e)}")
    
    async def _validate_instance_completeness(self, instance: ExtractionSchema, report: ValidationReport):
        """Validate instance completeness"""
        
        try:
            completeness_result = instance.validate_completeness()
            
            missing_required = completeness_result.get('missing_required', [])
            if missing_required:
                report.add_error(f"Missing required fields: {', '.join(missing_required)}")
            
            completeness_score = completeness_result.get('completeness_score', 0.0)
            if completeness_score < 0.8:
                report.add_warning(f"Low completeness score: {completeness_score:.2f}")
            
            report.completeness_score = completeness_score
            
        except Exception as e:
            report.add_error(f"Completeness validation failed: {str(e)}")
    
    async def _validate_data_quality(self, instance: ExtractionSchema, report: ValidationReport):
        """Validate data quality of instance"""
        
        if hasattr(instance, 'model_dump'):
            data = instance.model_dump()
        else:
            data = instance.__dict__
        
        # Check for empty or placeholder values
        empty_fields = []
        suspicious_values = []
        
        for field_name, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                empty_fields.append(field_name)
            elif isinstance(value, str):
                # Check for placeholder values
                placeholders = ['todo', 'tbd', 'unknown', 'n/a', 'null', 'none', '...']
                if value.lower().strip() in placeholders:
                    suspicious_values.append(field_name)
                # Check for very short values that might be incomplete
                elif len(value.strip()) < 3:
                    suspicious_values.append(field_name)
        
        if empty_fields:
            report.add_warning(f"Empty fields detected: {', '.join(empty_fields)}")
        
        if suspicious_values:
            report.add_warning(f"Suspicious/placeholder values in fields: {', '.join(suspicious_values)}")
    
    def _get_schema_fields(self, schema_class: Type[ExtractionSchema]) -> Dict[str, Any]:
        """Get field information from schema class"""
        if hasattr(schema_class, '__fields__'):
            return schema_class.__fields__
        elif hasattr(schema_class, 'model_fields'):
            return schema_class.model_fields
        else:
            return {}
    
    def _validate_field_type(self, field_name: str, field_type: Any) -> Dict[str, Any]:
        """Validate a field type definition"""
        
        result = {"valid": True, "message": ""}
        
        # Check for Any type (usually not ideal)
        if field_type == Any:
            result["valid"] = False
            result["message"] = "Consider using more specific type instead of 'Any'"
        
        # Check for overly complex nested types
        type_str = str(field_type)
        if type_str.count('[') > 3:  # Deeply nested generics
            result["valid"] = False
            result["message"] = "Type definition may be too complex"
        
        return result
    
    def _is_complex_type(self, field_type: Any) -> bool:
        """Check if a field type is considered complex"""
        if field_type is None:
            return False
        
        type_str = str(field_type)
        complex_indicators = ['List[', 'Dict[', 'Optional[', 'Union[', 'Tuple[']
        
        return any(indicator in type_str for indicator in complex_indicators)