"""
Structured extraction module for extracting information from repositories
using user-defined Pydantic models.
"""

from .structured_extractor import StructuredExtractor, ExtractionConfig

__all__ = [
    "StructuredExtractor",
    "ExtractionConfig"
]