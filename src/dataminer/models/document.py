# src/dataminer/models/document.py
"""Document and text-specific extraction models"""

from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field
from .base import BaseExtractedData, NestedExtractionSchema


class Reference(BaseModel):
    """Reference to external content"""
    title: str = Field(description="Reference title")
    url: Optional[str] = Field(None, description="Reference URL")
    type: Literal["link", "citation", "image", "video", "document", "code"] = Field(description="Reference type")
    description: Optional[str] = Field(None, description="Reference description")
    is_working: Optional[bool] = Field(None, description="Whether link is accessible")


class CodeBlock(BaseModel):
    """Code block within documentation"""
    language: Optional[str] = Field(None, description="Programming language")
    code: str = Field(description="Code content")
    caption: Optional[str] = Field(None, description="Code block caption")
    is_executable: bool = Field(default=False, description="Whether code can be executed")
    is_complete: bool = Field(default=True, description="Whether code is complete or snippet")
    line_numbers: bool = Field(default=False, description="Whether to show line numbers")


class Section(NestedExtractionSchema):
    """Document section"""
    
    # Section identification
    title: str = Field(description="Section title")
    level: int = Field(ge=1, le=6, description="Heading level (1-6)")
    section_id: Optional[str] = Field(None, description="Section ID or anchor")
    
    # Content
    content: str = Field(description="Section text content")
    summary: Optional[str] = Field(None, description="Section summary")
    
    # Structured content
    subsections: List[str] = Field(default_factory=list, description="IDs of subsections")
    code_blocks: List[CodeBlock] = Field(default_factory=list, description="Code blocks in section")
    references: List[Reference] = Field(default_factory=list, description="References in section")
    
    # Metadata
    word_count: int = Field(default=0, ge=0, description="Word count")
    reading_time_minutes: float = Field(default=0.0, ge=0.0, description="Estimated reading time")
    
    # Classification
    section_type: Literal["introduction", "overview", "tutorial", "reference", "example", "conclusion", "other"] = Field(default="other")
    topics: List[str] = Field(default_factory=list, description="Main topics covered")
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    
    def get_required_fields(self) -> List[str]:
        """Required fields for sections"""
        return super().get_required_fields() + ["title", "level", "content"]
    
    def get_confidence_fields(self) -> List[str]:
        """Fields that contribute to confidence scoring"""
        base_fields = super().get_confidence_fields()
        return base_fields + [
            "content", "summary", "code_blocks", "references", 
            "topics", "keywords", "word_count"
        ]
    
    def calculate_reading_time(self, words_per_minute: float = 200.0):
        """Calculate estimated reading time"""
        if not self.word_count:
            self.word_count = len(self.content.split())
        self.reading_time_minutes = self.word_count / words_per_minute


class Metadata(BaseModel):
    """Document metadata"""
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author name")
    created_date: Optional[str] = Field(None, description="Creation date")
    modified_date: Optional[str] = Field(None, description="Last modification date")
    version: Optional[str] = Field(None, description="Document version")
    
    # Classification
    document_type: Literal["readme", "tutorial", "api_doc", "guide", "specification", "other"] = Field(default="other")
    audience: Literal["beginner", "intermediate", "advanced", "expert", "mixed"] = Field(default="mixed")
    
    # Content metadata
    language: str = Field(default="en", description="Document language")
    word_count: int = Field(default=0, ge=0, description="Total word count")
    page_count: int = Field(default=1, ge=1, description="Number of pages")
    
    # Technical metadata
    format: Optional[str] = Field(None, description="Document format (md, rst, html, etc.)")
    encoding: str = Field(default="utf-8", description="Character encoding")
    
    # SEO and discoverability
    description: Optional[str] = Field(None, description="Document description")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    tags: List[str] = Field(default_factory=list, description="Document tags")


class DocumentStructure(BaseExtractedData):
    """Complete document structure"""
    
    # Document metadata
    metadata: Metadata = Field(default_factory=Metadata, description="Document metadata")
    
    # Content structure
    sections: List[Section] = Field(default_factory=list, description="Document sections")
    table_of_contents: List[Dict[str, Any]] = Field(default_factory=list, description="Table of contents")
    
    # Cross-references
    internal_links: List[Reference] = Field(default_factory=list, description="Internal document links")
    external_links: List[Reference] = Field(default_factory=list, description="External links")
    images: List[Reference] = Field(default_factory=list, description="Images and media")
    
    # Code content
    all_code_blocks: List[CodeBlock] = Field(default_factory=list, description="All code blocks")
    programming_languages: List[str] = Field(default_factory=list, description="Languages used in code blocks")
    
    # Document analysis
    readability_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Readability score")
    complexity_level: Literal["beginner", "intermediate", "advanced", "expert"] = Field(default="intermediate")
    completeness_indicators: List[str] = Field(default_factory=list, description="Indicators of document completeness")
    
    # Quality metrics
    has_introduction: bool = Field(default=False, description="Has introduction section")
    has_examples: bool = Field(default=False, description="Contains code examples")
    has_conclusion: bool = Field(default=False, description="Has conclusion section")
    link_health: float = Field(default=1.0, ge=0.0, le=1.0, description="Percentage of working links")
    
    def get_required_fields(self) -> List[str]:
        """Required fields for document structure"""
        return super().get_required_fields() + ["metadata", "sections"]
    
    def get_confidence_fields(self) -> List[str]:
        """Fields that contribute to confidence scoring"""
        base_fields = super().get_confidence_fields()
        return base_fields + [
            "sections", "table_of_contents", "all_code_blocks", 
            "programming_languages", "has_examples", "link_health"
        ]
    
    def extract_all_code_blocks(self):
        """Extract all code blocks from sections"""
        self.all_code_blocks = []
        languages = set()
        
        for section in self.sections:
            for code_block in section.code_blocks:
                self.all_code_blocks.append(code_block)
                if code_block.language:
                    languages.add(code_block.language)
        
        self.programming_languages = sorted(list(languages))
    
    def analyze_document_quality(self):
        """Analyze document quality indicators"""
        # Check for key sections
        section_titles = [s.title.lower() for s in self.sections]
        
        intro_keywords = ["introduction", "overview", "getting started", "about"]
        self.has_introduction = any(keyword in " ".join(section_titles) for keyword in intro_keywords)
        
        example_keywords = ["example", "tutorial", "how to", "usage"]
        self.has_examples = (
            any(keyword in " ".join(section_titles) for keyword in example_keywords) or
            len(self.all_code_blocks) > 0
        )
        
        conclusion_keywords = ["conclusion", "summary", "next steps", "further reading"]
        self.has_conclusion = any(keyword in " ".join(section_titles) for keyword in conclusion_keywords)
        
        # Analyze completeness
        self.completeness_indicators = []
        if self.has_introduction:
            self.completeness_indicators.append("Has introduction")
        if self.has_examples:
            self.completeness_indicators.append("Contains examples")
        if self.has_conclusion:
            self.completeness_indicators.append("Has conclusion")
        if len(self.sections) >= 3:
            self.completeness_indicators.append("Well-structured")
        if self.metadata.description:
            self.completeness_indicators.append("Has description")
    
    def get_section_by_title(self, title: str) -> Optional[Section]:
        """Find section by title"""
        for section in self.sections:
            if section.title.lower() == title.lower():
                return section
        return None
    
    def get_sections_by_type(self, section_type: str) -> List[Section]:
        """Get all sections of a specific type"""
        return [s for s in self.sections if s.section_type == section_type]


class FAQ(BaseModel):
    """Frequently Asked Questions"""
    question: str = Field(description="FAQ question")
    answer: str = Field(description="FAQ answer")
    category: Optional[str] = Field(None, description="FAQ category")
    tags: List[str] = Field(default_factory=list, description="Question tags")
    upvotes: int = Field(default=0, ge=0, description="Question upvotes")
    last_updated: Optional[str] = Field(None, description="Last update date")


class Glossary(BaseModel):
    """Glossary entry"""
    term: str = Field(description="Glossary term")
    definition: str = Field(description="Term definition")
    synonyms: List[str] = Field(default_factory=list, description="Alternative terms")
    related_terms: List[str] = Field(default_factory=list, description="Related terms")
    examples: List[str] = Field(default_factory=list, description="Usage examples")
    category: Optional[str] = Field(None, description="Term category")


class ChangelogEntry(BaseModel):
    """Changelog entry"""
    version: str = Field(description="Version number")
    date: Optional[str] = Field(None, description="Release date")
    changes: List[str] = Field(default_factory=list, description="List of changes")
    change_type: Literal["added", "changed", "deprecated", "removed", "fixed", "security"] = Field(description="Type of change")
    breaking_changes: bool = Field(default=False, description="Contains breaking changes")
    migration_notes: Optional[str] = Field(None, description="Migration notes")


class LicenseInfo(BaseModel):
    """License information"""
    name: str = Field(description="License name")
    spdx_id: Optional[str] = Field(None, description="SPDX license identifier")
    url: Optional[str] = Field(None, description="License URL")
    full_text: Optional[str] = Field(None, description="Full license text")
    permissions: List[str] = Field(default_factory=list, description="What the license permits")
    conditions: List[str] = Field(default_factory=list, description="License conditions")
    limitations: List[str] = Field(default_factory=list, description="License limitations")
    commercial_use: bool = Field(default=True, description="Allows commercial use")
    copyleft: bool = Field(default=False, description="Is a copyleft license")