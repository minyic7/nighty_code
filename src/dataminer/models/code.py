# src/dataminer/models/code.py
"""Code-specific extraction models"""

from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field
from .base import BaseExtractedData, NestedExtractionSchema


class FunctionSignature(BaseModel):
    """Function signature information"""
    name: str = Field(description="Function name")
    parameters: List[Dict[str, str]] = Field(default_factory=list, description="Parameters with types")
    return_type: Optional[str] = Field(None, description="Return type annotation")
    decorators: List[str] = Field(default_factory=list, description="Function decorators")
    is_async: bool = Field(default=False, description="Whether function is async")
    is_method: bool = Field(default=False, description="Whether this is a class method")
    is_static: bool = Field(default=False, description="Whether this is a static method")
    is_property: bool = Field(default=False, description="Whether this is a property")
    visibility: Literal["public", "private", "protected"] = Field(default="public")


class ClassDefinition(BaseModel):
    """Class definition information"""
    name: str = Field(description="Class name")
    base_classes: List[str] = Field(default_factory=list, description="Parent classes")
    methods: List[FunctionSignature] = Field(default_factory=list, description="Class methods")
    properties: List[str] = Field(default_factory=list, description="Class properties")
    decorators: List[str] = Field(default_factory=list, description="Class decorators")
    is_abstract: bool = Field(default=False, description="Whether class is abstract")
    visibility: Literal["public", "private", "protected"] = Field(default="public")
    docstring: Optional[str] = Field(None, description="Class docstring")


class CodeElement(NestedExtractionSchema):
    """Generic code element (function, class, variable, etc.)"""
    
    # Core identification
    name: str = Field(description="Element name")
    element_type: Literal["function", "class", "variable", "constant", "import", "module"] = Field(description="Type of code element")
    
    # Location information
    file_path: str = Field(description="File containing this element")
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")
    column_start: Optional[int] = Field(None, description="Starting column")
    column_end: Optional[int] = Field(None, description="Ending column")
    
    # Code content
    source_code: Optional[str] = Field(None, description="Raw source code")
    docstring: Optional[str] = Field(None, description="Documentation string")
    comments: List[str] = Field(default_factory=list, description="Associated comments")
    
    # Type and signature information
    signature: Optional[FunctionSignature] = Field(None, description="Function signature if applicable")
    class_definition: Optional[ClassDefinition] = Field(None, description="Class definition if applicable")
    data_type: Optional[str] = Field(None, description="Data type for variables")
    
    # Behavioral properties
    is_public: bool = Field(default=True, description="Whether element is public")
    is_deprecated: bool = Field(default=False, description="Whether element is deprecated")
    complexity_score: float = Field(default=0.0, ge=0.0, description="Cyclomatic complexity or similar")
    
    # Usage and relationships
    calls_made: List[str] = Field(default_factory=list, description="Functions/methods called by this element")
    called_by: List[str] = Field(default_factory=list, description="Elements that call this element")
    uses_variables: List[str] = Field(default_factory=list, description="Variables used by this element")
    modifies_variables: List[str] = Field(default_factory=list, description="Variables modified by this element")
    
    def get_required_fields(self) -> List[str]:
        """Required fields for code elements"""
        return super().get_required_fields() + ["name", "element_type", "file_path"]
    
    def get_confidence_fields(self) -> List[str]:
        """Fields that contribute to confidence scoring"""
        base_fields = super().get_confidence_fields()
        return base_fields + [
            "source_code", "docstring", "signature", "calls_made", 
            "line_start", "complexity_score"
        ]


class ModuleStructure(BaseExtractedData):
    """Structure of a code module/file"""
    
    # Module identification
    module_name: str = Field(description="Module name")
    file_path: str = Field(description="Full file path")
    package_path: Optional[str] = Field(None, description="Package path if applicable")
    
    # Content organization
    imports: List[Dict[str, str]] = Field(default_factory=list, description="Import statements")
    functions: List[CodeElement] = Field(default_factory=list, description="Module functions")
    classes: List[CodeElement] = Field(default_factory=list, description="Module classes") 
    variables: List[CodeElement] = Field(default_factory=list, description="Module variables")
    constants: List[CodeElement] = Field(default_factory=list, description="Module constants")
    
    # Module metadata
    module_docstring: Optional[str] = Field(None, description="Module-level docstring")
    author: Optional[str] = Field(None, description="Module author")
    license: Optional[str] = Field(None, description="License information")
    version: Optional[str] = Field(None, description="Module version")
    
    # Metrics
    lines_of_code: int = Field(default=0, ge=0, description="Total lines of code")
    comment_lines: int = Field(default=0, ge=0, description="Lines of comments")
    blank_lines: int = Field(default=0, ge=0, description="Blank lines")
    cyclomatic_complexity: float = Field(default=0.0, ge=0.0, description="Overall complexity")
    
    def get_required_fields(self) -> List[str]:
        """Required fields for module structure"""
        return super().get_required_fields() + ["module_name", "file_path"]
    
    def get_confidence_fields(self) -> List[str]:
        """Fields that contribute to confidence scoring"""
        base_fields = super().get_confidence_fields()
        return base_fields + [
            "imports", "functions", "classes", "module_docstring", 
            "lines_of_code", "cyclomatic_complexity"
        ]
    
    def get_all_elements(self) -> List[CodeElement]:
        """Get all code elements in the module"""
        return self.functions + self.classes + self.variables + self.constants
    
    def get_public_elements(self) -> List[CodeElement]:
        """Get only public code elements"""
        return [elem for elem in self.get_all_elements() if elem.is_public]
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        all_elements = self.get_all_elements()
        if all_elements:
            complexities = [elem.complexity_score for elem in all_elements if elem.complexity_score > 0]
            if complexities:
                self.cyclomatic_complexity = sum(complexities) / len(complexities)


class DependencyInfo(BaseModel):
    """Information about dependencies"""
    name: str = Field(description="Dependency name")
    version: Optional[str] = Field(None, description="Version specification")
    dependency_type: Literal["internal", "external", "standard_library"] = Field(description="Type of dependency")
    import_path: str = Field(description="Import path used")
    used_symbols: List[str] = Field(default_factory=list, description="Specific symbols used")
    usage_count: int = Field(default=0, ge=0, description="Number of times used")
    is_optional: bool = Field(default=False, description="Whether dependency is optional")


class CodeMetrics(BaseModel):
    """Code quality and complexity metrics"""
    
    # Basic metrics
    lines_of_code: int = Field(ge=0, description="Lines of code")
    lines_of_comments: int = Field(ge=0, description="Lines of comments")
    blank_lines: int = Field(ge=0, description="Blank lines")
    
    # Complexity metrics
    cyclomatic_complexity: float = Field(ge=0.0, description="Cyclomatic complexity")
    halstead_complexity: Optional[float] = Field(None, ge=0.0, description="Halstead complexity")
    maintainability_index: Optional[float] = Field(None, ge=0.0, le=100.0, description="Maintainability index")
    
    # Structure metrics
    function_count: int = Field(ge=0, description="Number of functions")
    class_count: int = Field(ge=0, description="Number of classes") 
    max_nesting_depth: int = Field(ge=0, description="Maximum nesting depth")
    
    # Quality indicators
    comment_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of comments to code")
    test_coverage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Test coverage percentage")
    duplication_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Code duplication percentage")
    
    # Dependencies
    dependency_count: int = Field(ge=0, description="Number of dependencies")
    internal_dependencies: int = Field(ge=0, description="Internal dependencies")
    external_dependencies: int = Field(ge=0, description="External dependencies")
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score (0-1)"""
        factors = []
        
        # Comment ratio (higher is better, up to 0.3)
        comment_factor = min(self.comment_ratio / 0.3, 1.0)
        factors.append(comment_factor * 0.2)
        
        # Complexity (lower is better)
        complexity_factor = max(0, 1.0 - (self.cyclomatic_complexity - 1) / 10)
        factors.append(complexity_factor * 0.3)
        
        # Maintainability index (if available)
        if self.maintainability_index is not None:
            maintainability_factor = self.maintainability_index / 100.0
            factors.append(maintainability_factor * 0.3)
        
        # Test coverage (if available)
        if self.test_coverage is not None:
            coverage_factor = self.test_coverage / 100.0
            factors.append(coverage_factor * 0.2)
        
        return sum(factors) / len(factors) if factors else 0.5


class TestInfo(BaseModel):
    """Information about tests"""
    test_name: str = Field(description="Test name or description")
    test_type: Literal["unit", "integration", "functional", "performance", "other"] = Field(description="Type of test")
    file_path: str = Field(description="Test file path")
    target_elements: List[str] = Field(default_factory=list, description="Code elements being tested")
    assertions_count: int = Field(default=0, ge=0, description="Number of assertions")
    is_passing: Optional[bool] = Field(None, description="Whether test passes")
    execution_time_ms: Optional[float] = Field(None, ge=0.0, description="Test execution time")


class APIInfo(BaseModel):
    """Information about API endpoints or interfaces"""
    name: str = Field(description="API name")
    endpoint_path: Optional[str] = Field(None, description="Endpoint path")
    http_method: Optional[str] = Field(None, description="HTTP method")
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="API parameters")
    return_type: Optional[str] = Field(None, description="Return type")
    status_codes: List[int] = Field(default_factory=list, description="Possible HTTP status codes")
    authentication_required: bool = Field(default=False, description="Whether auth is required")
    rate_limited: bool = Field(default=False, description="Whether rate limited")
    deprecated: bool = Field(default=False, description="Whether deprecated")
    documentation: Optional[str] = Field(None, description="API documentation")