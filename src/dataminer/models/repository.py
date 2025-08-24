# src/dataminer/models/repository.py
"""Repository-level extraction models"""

from typing import Dict, List, Optional, Union, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
from .base import BaseExtractedData
from .code import ModuleStructure, CodeMetrics, DependencyInfo, APIInfo, TestInfo
from .document import DocumentStructure, LicenseInfo, ChangelogEntry


class FileInfo(BaseModel):
    """Information about a file in the repository"""
    path: str = Field(description="Relative file path")
    name: str = Field(description="File name")
    extension: str = Field(description="File extension")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    
    # Classification
    file_type: Literal["code", "documentation", "config", "data", "test", "asset", "other"] = Field(description="File type classification")
    language: Optional[str] = Field(None, description="Programming language")
    
    # Content metadata
    encoding: str = Field(default="utf-8", description="File encoding")
    line_count: int = Field(default=0, ge=0, description="Number of lines")
    is_binary: bool = Field(default=False, description="Whether file is binary")
    
    # Analysis metadata
    complexity_score: float = Field(default=0.0, ge=0.0, description="File complexity score")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance in repository")
    last_modified: Optional[datetime] = None
    
    # Content hashes for change detection
    content_hash: Optional[str] = Field(None, description="Content hash")
    structure_hash: Optional[str] = Field(None, description="Structure hash")


class DirectoryStructure(BaseModel):
    """Directory structure information"""
    path: str = Field(description="Directory path")
    name: str = Field(description="Directory name")
    
    # Contents
    files: List[FileInfo] = Field(default_factory=list, description="Files in directory")
    subdirectories: List[str] = Field(default_factory=list, description="Subdirectory names")
    
    # Metrics
    total_files: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    file_type_counts: Dict[str, int] = Field(default_factory=dict)
    language_counts: Dict[str, int] = Field(default_factory=dict)
    
    # Classification
    directory_type: Literal["source", "test", "docs", "config", "assets", "build", "other"] = Field(default="other")
    is_package: bool = Field(default=False, description="Whether directory is a package/module")


class TechnologyStack(BaseModel):
    """Technology stack analysis"""
    
    # Primary technologies
    primary_language: Optional[str] = Field(None, description="Main programming language")
    languages: Dict[str, float] = Field(default_factory=dict, description="Languages and usage percentages")
    
    # Frameworks and libraries
    frameworks: List[str] = Field(default_factory=list, description="Detected frameworks")
    libraries: List[DependencyInfo] = Field(default_factory=list, description="External libraries")
    
    # Development tools
    build_tools: List[str] = Field(default_factory=list, description="Build tools (make, npm, etc.)")
    test_frameworks: List[str] = Field(default_factory=list, description="Testing frameworks")
    linting_tools: List[str] = Field(default_factory=list, description="Code quality tools")
    
    # Infrastructure
    deployment_tools: List[str] = Field(default_factory=list, description="Deployment/DevOps tools")
    databases: List[str] = Field(default_factory=list, description="Database technologies")
    cloud_services: List[str] = Field(default_factory=list, description="Cloud service providers")
    
    # Confidence scores
    detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in detection")
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How complete the analysis is")


class ProjectMetadata(BaseModel):
    """High-level project metadata"""
    
    # Basic information
    name: str = Field(description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    version: Optional[str] = Field(None, description="Current version")
    
    # People and organization
    authors: List[str] = Field(default_factory=list, description="Project authors")
    maintainers: List[str] = Field(default_factory=list, description="Current maintainers")
    contributors_count: Optional[int] = Field(None, ge=0, description="Number of contributors")
    
    # Project classification
    project_type: Literal["library", "application", "framework", "tool", "documentation", "other"] = Field(default="other")
    maturity_level: Literal["experimental", "alpha", "beta", "stable", "mature", "deprecated"] = Field(default="stable")
    
    # URLs and contacts
    homepage: Optional[str] = Field(None, description="Project homepage")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    issue_tracker: Optional[str] = Field(None, description="Issue tracker URL")
    
    # Licensing
    license: Optional[LicenseInfo] = Field(None, description="License information")
    copyright_holders: List[str] = Field(default_factory=list, description="Copyright holders")
    
    # Activity metrics
    last_commit_date: Optional[datetime] = None
    creation_date: Optional[datetime] = None
    is_active: bool = Field(default=True, description="Whether project is actively maintained")
    
    # Quality indicators
    has_tests: bool = Field(default=False, description="Has test suite")
    has_ci: bool = Field(default=False, description="Has continuous integration")
    has_documentation: bool = Field(default=False, description="Has documentation")
    has_license: bool = Field(default=False, description="Has license file")
    has_readme: bool = Field(default=False, description="Has README file")


class RepositoryMap(BaseExtractedData):
    """Complete repository structure and analysis"""
    
    # Repository identification
    repository_path: str = Field(description="Root path of repository")
    repository_name: str = Field(description="Repository name")
    
    # High-level structure
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata, description="Project metadata")
    technology_stack: TechnologyStack = Field(default_factory=TechnologyStack, description="Technology stack")
    
    # Directory structure
    root_directory: DirectoryStructure = Field(description="Root directory structure")
    important_directories: Dict[str, DirectoryStructure] = Field(default_factory=dict, description="Key directories")
    
    # File organization
    all_files: List[FileInfo] = Field(default_factory=list, description="All files in repository")
    code_files: List[FileInfo] = Field(default_factory=list, description="Source code files")
    test_files: List[FileInfo] = Field(default_factory=list, description="Test files")
    documentation_files: List[FileInfo] = Field(default_factory=list, description="Documentation files")
    config_files: List[FileInfo] = Field(default_factory=list, description="Configuration files")
    
    # Code structure
    modules: List[ModuleStructure] = Field(default_factory=list, description="Analyzed modules")
    apis: List[APIInfo] = Field(default_factory=list, description="Detected APIs")
    tests: List[TestInfo] = Field(default_factory=list, description="Test information")
    
    # Documentation
    readme_content: Optional[DocumentStructure] = Field(None, description="README document structure")
    changelog: List[ChangelogEntry] = Field(default_factory=list, description="Changelog entries")
    additional_docs: List[DocumentStructure] = Field(default_factory=list, description="Other documentation")
    
    # Dependencies and relationships
    dependencies: List[DependencyInfo] = Field(default_factory=list, description="Project dependencies")
    internal_dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Internal module dependencies")
    
    # Metrics and analysis
    repository_metrics: CodeMetrics = Field(default_factory=CodeMetrics, description="Overall repository metrics")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall quality score")
    complexity_distribution: Dict[str, int] = Field(default_factory=dict, description="Complexity distribution")
    
    # Health indicators
    health_indicators: List[str] = Field(default_factory=list, description="Repository health indicators")
    potential_issues: List[str] = Field(default_factory=list, description="Potential issues identified")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    def get_required_fields(self) -> List[str]:
        """Required fields for repository map"""
        return super().get_required_fields() + [
            "repository_path", "repository_name", "root_directory", "all_files"
        ]
    
    def get_confidence_fields(self) -> List[str]:
        """Fields that contribute to confidence scoring"""
        base_fields = super().get_confidence_fields()
        return base_fields + [
            "technology_stack", "modules", "dependencies", 
            "repository_metrics", "health_indicators", "all_files"
        ]
    
    def analyze_repository_health(self):
        """Analyze repository health and provide recommendations"""
        self.health_indicators = []
        self.potential_issues = []
        self.recommendations = []
        
        # Analyze structure
        if self.project_metadata.has_readme:
            self.health_indicators.append("Has README file")
        else:
            self.potential_issues.append("Missing README file")
            self.recommendations.append("Add a comprehensive README file")
        
        if self.project_metadata.has_license:
            self.health_indicators.append("Has license")
        else:
            self.potential_issues.append("No license file")
            self.recommendations.append("Add appropriate license file")
        
        if self.project_metadata.has_tests:
            self.health_indicators.append("Has test suite")
        else:
            self.potential_issues.append("No tests found")
            self.recommendations.append("Add unit tests")
        
        # Analyze code quality
        if self.repository_metrics.comment_ratio > 0.1:
            self.health_indicators.append("Good code documentation")
        else:
            self.potential_issues.append("Low comment ratio")
            self.recommendations.append("Add more code comments and documentation")
        
        # Analyze dependencies
        external_deps = len([d for d in self.dependencies if d.dependency_type == "external"])
        if external_deps < 20:
            self.health_indicators.append("Manageable dependency count")
        else:
            self.potential_issues.append("High number of external dependencies")
            self.recommendations.append("Review and reduce unnecessary dependencies")
        
        # Calculate quality score
        positive_factors = len(self.health_indicators)
        negative_factors = len(self.potential_issues)
        total_factors = positive_factors + negative_factors
        
        if total_factors > 0:
            self.quality_score = positive_factors / total_factors
        else:
            self.quality_score = 0.5
    
    def get_files_by_type(self, file_type: str) -> List[FileInfo]:
        """Get all files of a specific type"""
        return [f for f in self.all_files if f.file_type == file_type]
    
    def get_files_by_language(self, language: str) -> List[FileInfo]:
        """Get all files of a specific language"""
        return [f for f in self.all_files if f.language == language]
    
    def calculate_repository_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive repository statistics"""
        stats = {
            "total_files": len(self.all_files),
            "total_size_bytes": sum(f.size_bytes for f in self.all_files),
            "code_files": len(self.code_files),
            "test_files": len(self.test_files),
            "documentation_files": len(self.documentation_files),
            "languages": {},
            "file_types": {},
            "complexity_stats": {
                "mean": 0.0,
                "median": 0.0,
                "max": 0.0
            }
        }
        
        # Language distribution
        for file in self.all_files:
            if file.language:
                stats["languages"][file.language] = stats["languages"].get(file.language, 0) + 1
        
        # File type distribution
        for file in self.all_files:
            stats["file_types"][file.file_type] = stats["file_types"].get(file.file_type, 0) + 1
        
        # Complexity statistics
        complexity_scores = [f.complexity_score for f in self.all_files if f.complexity_score > 0]
        if complexity_scores:
            stats["complexity_stats"]["mean"] = sum(complexity_scores) / len(complexity_scores)
            stats["complexity_stats"]["median"] = sorted(complexity_scores)[len(complexity_scores) // 2]
            stats["complexity_stats"]["max"] = max(complexity_scores)
        
        return stats


class RepositoryComparison(BaseModel):
    """Comparison between two repository states"""
    
    baseline_repo: str = Field(description="Baseline repository identifier")
    comparison_repo: str = Field(description="Comparison repository identifier")
    
    # File changes
    added_files: List[str] = Field(default_factory=list, description="Newly added files")
    removed_files: List[str] = Field(default_factory=list, description="Removed files")
    modified_files: List[str] = Field(default_factory=list, description="Modified files")
    
    # Metric changes
    size_change_bytes: int = Field(default=0, description="Change in repository size")
    complexity_change: float = Field(default=0.0, description="Change in overall complexity")
    quality_score_change: float = Field(default=0.0, description="Change in quality score")
    
    # Dependency changes
    added_dependencies: List[str] = Field(default_factory=list)
    removed_dependencies: List[str] = Field(default_factory=list)
    updated_dependencies: List[str] = Field(default_factory=list)
    
    # Summary
    change_summary: str = Field(description="Human-readable summary of changes")
    significant_changes: List[str] = Field(default_factory=list, description="Most significant changes")
    impact_assessment: Literal["low", "medium", "high", "critical"] = Field(default="low")