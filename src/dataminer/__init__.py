"""DataMiner module - Production-ready data extraction with LLM, MCP, and Copilot integration"""

from .core import *
from .client import DataMinerClient
from .models import *
from .strategies import *
from .utils import *

__version__ = "1.0.0"

__all__ = [
    "DataMinerClient",
    # Core types
    "ExtractionConfig",
    "ExtractionRequest", 
    "ExtractionResult",
    "ConfidenceMetrics",
    "GapAnalysis",
    # Models
    "ExtractionSchema",
    "CodeElement",
    "DocumentStructure",
    "RepositoryMap",
    # Strategies
    "BaseExtractionStrategy",
    "SimpleExtractionStrategy",
    "MultiStageExtractionStrategy",
    "CognitiveExtractionStrategy",
    # Utils
    "SchemaValidator",
    "RepositoryAnalyzer",
]