"""
File classification module.

This module provides file type detection and classification capabilities
using multiple detection strategies including extension-based, content-based,
and pattern-based classification.
"""

from .rules import ClassificationRules
from .detectors import (
    ExtensionDetector,
    ContentDetector, 
    ShebangDetector,
    FilenameDetector,
    FrameworkDetector,
    HeuristicDetector
)
from .metrics import MetricsCalculator

__all__ = [
    'ClassificationRules',
    'ExtensionDetector',
    'ContentDetector',
    'ShebangDetector', 
    'FilenameDetector',
    'FrameworkDetector',
    'HeuristicDetector',
    'MetricsCalculator'
]