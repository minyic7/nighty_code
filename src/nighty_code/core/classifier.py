"""
File classification system for determining file types and characteristics.

This module provides the main FileClassifier class that orchestrates
multiple detection strategies to accurately classify files.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from .models import (
    FileType, 
    Framework,
    ClassificationResult, 
    FileClassification, 
    FileMetrics,
    ContentSignature
)
from .classification import (
    ClassificationRules,
    ExtensionDetector,
    FilenameDetector,
    ShebangDetector,
    ContentDetector,
    FrameworkDetector,
    HeuristicDetector,
    MetricsCalculator
)


logger = logging.getLogger(__name__)


class FileClassifier:
    """
    Main file classification system.
    
    Uses multiple detection strategies in order of reliability:
    1. Shebang detection (highest confidence)
    2. Extension detection (high confidence)
    3. Filename pattern detection (medium-high confidence)
    4. Content analysis (medium confidence)
    5. Heuristic detection (fallback)
    """
    
    def __init__(self):
        self.rules = ClassificationRules()
        self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize all detection strategies."""
        self.extension_detector = ExtensionDetector(self.rules)
        self.filename_detector = FilenameDetector(self.rules)
        self.shebang_detector = ShebangDetector(self.rules)
        self.content_detector = ContentDetector(self.rules)
        self.framework_detector = FrameworkDetector(self.rules)
        self.heuristic_detector = HeuristicDetector(self.rules)
        self.metrics_calculator = MetricsCalculator(self.rules)
    
    def classify_file(
        self, 
        file_path: Path, 
        content: Optional[str] = None,
        read_content: bool = True
    ) -> FileClassification:
        """
        Classify a file and return comprehensive classification information.
        
        Args:
            file_path: Path to the file to classify
            content: Optional file content (if not provided, will read from file)
            read_content: Whether to read file content if not provided
            
        Returns:
            FileClassification with complete analysis results
        """
        try:
            # Read content if needed and possible
            if content is None and read_content:
                content = self._read_file_content(file_path)
            
            # Run classification
            classification_result = self._classify_file_type(file_path, content)
            
            # Calculate metrics
            metrics = self._calculate_metrics(content or "", classification_result.file_type)
            
            # Generate content signature
            content_signature = self._generate_content_signature(
                content or "", classification_result.file_type
            )
            
            # Detect frameworks
            if content:
                frameworks = self.framework_detector.detect_frameworks(content)
                for framework in frameworks:
                    classification_result.add_framework(framework)
            
            # Create final classification
            file_classification = FileClassification(
                file_path=file_path,
                classification_result=classification_result,
                metrics=metrics,
                content_signature=content_signature
            )
            
            logger.debug(f"Classified {file_path} as {classification_result.file_type} "
                        f"with confidence {classification_result.confidence:.2f}")
            
            return file_classification
            
        except Exception as e:
            logger.error(f"Error classifying file {file_path}: {e}")
            
            # Return minimal classification on error
            error_result = ClassificationResult(
                file_path=file_path,
                file_type=FileType.UNKNOWN,
                confidence=0.0
            )
            
            error_metrics = FileMetrics(
                size_bytes=0,
                line_count=0,
                non_empty_lines=0
            )
            
            error_signature = ContentSignature(
                content_hash="",
                structural_hash=""
            )
            
            error_classification = FileClassification(
                file_path=file_path,
                classification_result=error_result,
                metrics=error_metrics,
                content_signature=error_signature
            )
            error_classification.add_error(str(e))
            
            return error_classification
    
    def _classify_file_type(self, file_path: Path, content: Optional[str]) -> ClassificationResult:
        """Run the classification detection cascade."""
        
        # Try each detector in order of confidence
        detectors = [
            ("shebang", self.shebang_detector),
            ("extension", self.extension_detector), 
            ("filename", self.filename_detector),
            ("content", self.content_detector),
            ("heuristic", self.heuristic_detector),
        ]
        
        best_result = None
        detection_methods = []
        
        for detector_name, detector in detectors:
            try:
                result = detector.detect(file_path, content)
                if result:
                    file_type, confidence = result
                    
                    # Keep track of what detected it
                    detection_methods.append(detector_name)
                    
                    # Use the first detector that gives a confident result
                    if not best_result or confidence > best_result[1]:
                        best_result = (file_type, confidence, detector_name)
                    
                    # If we have high confidence, we can stop
                    if confidence >= 0.9:
                        break
                        
            except Exception as e:
                logger.warning(f"Error in {detector_name} detector for {file_path}: {e}")
                continue
        
        if best_result:
            file_type, confidence, primary_detector = best_result
            
            # Create classification result
            classification_result = ClassificationResult(
                file_path=file_path,
                file_type=file_type,
                confidence=confidence
            )
            
            # Set detection method flags
            classification_result.detected_by_extension = "extension" in detection_methods
            classification_result.detected_by_content = "content" in detection_methods
            classification_result.detected_by_filename = "filename" in detection_methods
            classification_result.detected_by_shebang = "shebang" in detection_methods
            
            return classification_result
        
        else:
            # Fallback to unknown
            return ClassificationResult(
                file_path=file_path,
                file_type=FileType.UNKNOWN,
                confidence=0.0
            )
    
    def _calculate_metrics(self, content: str, file_type: FileType) -> FileMetrics:
        """Calculate file metrics."""
        return self.metrics_calculator.calculate_metrics(content, file_type)
    
    def _generate_content_signature(self, content: str, file_type: FileType) -> ContentSignature:
        """Generate content signature."""
        return ContentSignature.from_content(content, file_type)
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, treat as binary
            logger.warning(f"Could not decode {file_path} as text")
            return None
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def classify_files(
        self, 
        file_paths: List[Path], 
        read_content: bool = True
    ) -> List[FileClassification]:
        """
        Classify multiple files efficiently.
        
        Args:
            file_paths: List of file paths to classify
            read_content: Whether to read file contents
            
        Returns:
            List of FileClassification results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.classify_file(file_path, read_content=read_content)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying {file_path}: {e}")
                # Continue with other files
                continue
        
        return results
    
    def get_classification_summary(
        self, 
        classifications: List[FileClassification]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from classification results.
        
        Args:
            classifications: List of file classifications
            
        Returns:
            Dictionary with summary statistics
        """
        if not classifications:
            return {}
        
        # Count by file type
        type_counts = {}
        framework_counts = {}
        total_files = len(classifications)
        total_size = 0
        confidence_scores = []
        
        for classification in classifications:
            file_type = classification.classification_result.file_type
            type_counts[file_type.value] = type_counts.get(file_type.value, 0) + 1
            
            # Count frameworks
            for framework in classification.classification_result.frameworks:
                framework_counts[framework.value] = framework_counts.get(framework.value, 0) + 1
            
            total_size += classification.metrics.size_bytes
            confidence_scores.append(classification.classification_result.confidence)
        
        # Calculate averages
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_file_size = total_size / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'average_file_size_bytes': avg_file_size,
            'average_confidence': avg_confidence,
            'file_type_distribution': type_counts,
            'framework_distribution': framework_counts,
            'high_confidence_files': len([c for c in confidence_scores if c >= 0.8]),
            'low_confidence_files': len([c for c in confidence_scores if c < 0.5]),
        }


def classify_file(file_path: Path, content: Optional[str] = None) -> FileClassification:
    """
    Convenience function to classify a single file.
    
    Args:
        file_path: Path to the file
        content: Optional file content
        
    Returns:
        FileClassification result
    """
    classifier = FileClassifier()
    return classifier.classify_file(file_path, content)