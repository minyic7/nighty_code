# src/dataminer/client.py
"""Main DataMiner client for orchestrating extraction workflows"""

from typing import Dict, List, Optional, Any, Union, Type, TypeVar, Generic
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import hashlib
import json
import pickle

from .core.types import (
    ExtractionRequest, ExtractionResult, ExtractionConfig,
    ProcessingMode, ExtractionSession, ProgressCallback, ConfidenceMetrics
)
from .core.config import DataMinerConfig, create_default_config
from .core.exceptions import DataMinerError, ConfigurationError, IntegrationError
from .models.base import ExtractionSchema
from .strategies.base import BaseExtractionStrategy
from .strategies.simple import SimpleExtractionStrategy
from .strategies.multistage import MultiStageExtractionStrategy
from .strategies.cognitive import CognitiveExtractionStrategy
from .utils.repository import RepositoryAnalyzer
from .utils.validator import SchemaValidator

T = TypeVar('T', bound=ExtractionSchema)

logger = logging.getLogger(__name__)


class DataMinerClient:
    """Main client for data extraction operations"""
    
    def __init__(self, config: Optional[DataMinerConfig] = None):
        """Initialize DataMiner client"""
        self.config = config or create_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.strategies: Dict[str, BaseExtractionStrategy] = {}
        self._register_strategies()
        
        # Session management
        self.sessions: Dict[str, ExtractionSession] = {}
        self.current_session: Optional[ExtractionSession] = None
        
        # Caching
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.total_extractions = 0
        self.total_processing_time = 0.0
        self.extraction_history: List[Dict[str, Any]] = []
        
        # Integration clients
        self.llm_manager = None
        self.mcp_server = None
        self.copilot_workflow = None
        
        # Utilities
        self.repository_analyzer = RepositoryAnalyzer()
        self.schema_validator = SchemaValidator()
        
        # Setup logging
        self._setup_logging()
    
    async def initialize(self) -> None:
        """Initialize the DataMiner client and its integrations"""
        try:
            self.logger.info("Initializing DataMiner client")
            
            # Validate configuration
            config_issues = self.config.validate()
            if config_issues:
                raise ConfigurationError(f"Configuration validation failed: {', '.join(config_issues)}")
            
            # Initialize LLM integration
            await self._initialize_llm()
            
            # Initialize MCP integration
            if self.config.extraction.enable_mcp_tools:
                await self._initialize_mcp()
            
            # Initialize Copilot integration
            if self.config.extraction.use_copilot_reasoning:
                await self._initialize_copilot()
            
            # Create cache directory
            if self.config.cache_directory:
                self.config.cache_directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("DataMiner client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DataMiner client: {e}")
            raise DataMinerError(f"Initialization failed: {e}")
    
    async def extract(
        self,
        content: Union[str, List[str], Path],
        schema: Type[T],
        mode: Optional[ProcessingMode] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> ExtractionResult[T]:
        """
        Extract structured data from content
        
        Args:
            content: Content to extract from (text, list of texts, or file path)
            schema: Pydantic model defining the target schema
            mode: Processing mode (defaults to config default)
            config_overrides: Override configuration settings
            session_id: Session ID for tracking related extractions
            progress_callback: Callback for progress updates
            
        Returns:
            ExtractionResult with extracted data and metadata
        """
        try:
            # Prepare extraction request
            request = await self._prepare_extraction_request(
                content, schema, mode, config_overrides
            )
            
            # Check cache first
            if self.config.enable_result_cache:
                cached_result = await self._check_cache(request)
                if cached_result:
                    self.cache_hits += 1
                    self.logger.info(f"Cache hit for request {request.request_id}")
                    return cached_result
                self.cache_misses += 1
            
            # Get or create session
            session = await self._get_or_create_session(session_id)
            session.add_request(request.request_id)
            
            # Select and execute strategy
            strategy = self._select_strategy(request.mode)
            
            if progress_callback:
                strategy.add_progress_callback(progress_callback)
            
            # Execute extraction
            self.logger.info(f"Starting extraction with {strategy.name} strategy")
            start_time = datetime.now()
            
            result = await strategy.extract(request, self.config.extraction)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update statistics
            self.total_extractions += 1
            self.total_processing_time += processing_time
            
            # Update session
            session.update_stats(result)
            
            # Cache result
            if self.config.enable_result_cache and result.success:
                await self._cache_result(request, result)
            
            # Record extraction
            self.extraction_history.append({
                "request_id": request.request_id,
                "schema": schema.__name__,
                "mode": request.mode.value,
                "success": result.success,
                "confidence": result.confidence.overall,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": datetime.now()
            })
            
            self.logger.info(f"Extraction completed: success={result.success}, confidence={result.confidence.overall:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}", exc_info=True)
            raise DataMinerError(f"Extraction failed: {e}")
    
    async def extract_from_repository(
        self,
        repository_path: Path,
        schema: Type[T],
        file_patterns: Optional[List[str]] = None,
        mode: Optional[ProcessingMode] = None,
        max_files: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> ExtractionResult[T]:
        """
        Extract data from an entire repository
        
        Args:
            repository_path: Path to repository root
            schema: Target extraction schema
            file_patterns: File patterns to include (e.g., ["*.py", "*.md"])
            mode: Processing mode
            max_files: Maximum number of files to process
            progress_callback: Progress callback
            
        Returns:
            ExtractionResult with repository-wide extraction
        """
        try:
            self.logger.info(f"Starting repository extraction from {repository_path}")
            
            # Analyze repository structure
            repo_analysis = await self.repository_analyzer.analyze_repository(
                repository_path,
                file_patterns=file_patterns,
                max_files=max_files
            )
            
            if not repo_analysis.recommended_files:
                raise DataMinerError("No suitable files found in repository")
            
            # Read content from recommended files
            content_items = []
            for file_path in repo_analysis.recommended_files[:max_files or 50]:
                try:
                    content = await self._read_file_content(file_path)
                    content_items.append(f"File: {file_path}\n\n{content}")
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_path}: {e}")
            
            if not content_items:
                raise DataMinerError("No content could be read from repository")
            
            # Create extraction request with repository context
            context = {
                "repository_path": str(repository_path),
                "repository_structure": repo_analysis.structure_summary,
                "total_files": len(repo_analysis.recommended_files),
                "languages": repo_analysis.languages
            }
            
            return await self.extract(
                content=content_items,
                schema=schema,
                mode=mode,
                config_overrides={
                    "file_patterns": file_patterns,
                    "max_files": max_files,
                    "context": context
                },
                progress_callback=progress_callback
            )
            
        except Exception as e:
            self.logger.error(f"Repository extraction failed: {e}")
            raise DataMinerError(f"Repository extraction failed: {e}")
    
    async def batch_extract(
        self,
        requests: List[tuple[Union[str, List[str]], Type[ExtractionSchema]]],
        mode: Optional[ProcessingMode] = None,
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> List[ExtractionResult]:
        """
        Perform batch extraction on multiple content/schema pairs
        
        Args:
            requests: List of (content, schema) tuples
            mode: Processing mode for all extractions
            max_concurrent: Maximum concurrent extractions
            progress_callback: Progress callback
            
        Returns:
            List of ExtractionResults
        """
        try:
            max_concurrent = max_concurrent or self.config.extraction.max_concurrent_extractions
            
            self.logger.info(f"Starting batch extraction of {len(requests)} items")
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def extract_single(content_schema_pair):
                content, schema = content_schema_pair
                async with semaphore:
                    return await self.extract(
                        content=content,
                        schema=schema,
                        mode=mode,
                        progress_callback=progress_callback
                    )
            
            # Execute all extractions concurrently
            results = await asyncio.gather(
                *[extract_single(pair) for pair in requests],
                return_exceptions=True
            )
            
            # Convert exceptions to failed results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create failed result without generic type parameter
                    content, schema = requests[i]
                    # Create a basic ExtractionResult without the generic type
                    failed_result = ExtractionResult(
                        data=None,
                        confidence=ConfidenceMetrics(overall=self.config.extraction.min_confidence_threshold),
                        request_id=f"batch_failed_{i}",
                        errors=[f"Batch extraction failed: {str(result)}"],
                        completed_at=datetime.now()
                    )
                    processed_results.append(failed_result)
                else:
                    processed_results.append(result)
            
            successful = sum(1 for r in processed_results if r.success)
            self.logger.info(f"Batch extraction completed: {successful}/{len(requests)} successful")
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Batch extraction failed: {e}")
            raise DataMinerError(f"Batch extraction failed: {e}")
    
    async def validate_schema(self, schema: Type[ExtractionSchema]) -> Dict[str, Any]:
        """Validate an extraction schema"""
        return await self.schema_validator.validate_schema(schema)
    
    async def get_extraction_capabilities(self) -> Dict[str, Any]:
        """Get information about extraction capabilities"""
        return {
            "strategies": list(self.strategies.keys()),
            "supported_modes": [mode.value for mode in ProcessingMode],
            "integrations": {
                "llm_available": self.llm_manager is not None,
                "mcp_available": self.mcp_server is not None,
                "copilot_available": self.copilot_workflow is not None
            },
            "configuration": {
                "max_concurrent_extractions": self.config.extraction.max_concurrent_extractions,
                "chunk_size": self.config.extraction.chunk_size,
                "min_confidence_threshold": self.config.extraction.min_confidence_threshold,
                "supported_languages": self.config.extraction.supported_languages
            },
            "statistics": {
                "total_extractions": self.total_extractions,
                "total_processing_time": self.total_processing_time,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
                "average_processing_time": self.total_processing_time / self.total_extractions if self.total_extractions > 0 else 0.0
            }
        }
    
    async def cleanup(self):
        """Cleanup resources and save state"""
        try:
            self.logger.info("Cleaning up DataMiner client")
            
            # Clear cache if needed
            if not self.config.enable_result_cache:
                self.cache.clear()
            
            # Close integration connections
            if self.mcp_server:
                try:
                    await self.mcp_server.cleanup()
                except:
                    pass
            
            # Save extraction history if configured
            if self.config.log_file and self.extraction_history:
                history_file = self.config.log_file.parent / "extraction_history.json"
                with open(history_file, 'w') as f:
                    json.dump(self.extraction_history, f, default=str, indent=2)
            
            self.logger.info("DataMiner client cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def _register_strategies(self):
        """Register available extraction strategies"""
        strategies = [
            SimpleExtractionStrategy(),
            MultiStageExtractionStrategy(),
            CognitiveExtractionStrategy()
        ]
        
        for strategy in strategies:
            self.strategies[strategy.name] = strategy
    
    def _select_strategy(self, mode: ProcessingMode) -> BaseExtractionStrategy:
        """Select appropriate strategy for processing mode"""
        # Find strategies that support the mode
        compatible_strategies = [
            strategy for strategy in self.strategies.values()
            if strategy.supports_mode(mode)
        ]
        
        if not compatible_strategies:
            # Fallback to simple strategy
            return self.strategies["SimpleExtraction"]
        
        # Select best strategy for mode
        strategy_preferences = {
            ProcessingMode.FAST: "SimpleExtraction",
            ProcessingMode.THOROUGH: "MultiStageExtraction", 
            ProcessingMode.COGNITIVE: "CognitiveExtraction",
            ProcessingMode.HYBRID: "MultiStageExtraction"  # Prefer multi-stage for hybrid
        }
        
        preferred = strategy_preferences.get(mode)
        if preferred and preferred in self.strategies:
            strategy = self.strategies[preferred]
            if strategy.supports_mode(mode):
                return strategy
        
        # Return first compatible strategy
        return compatible_strategies[0]
    
    async def _prepare_extraction_request(
        self,
        content: Union[str, List[str], Path],
        schema: Type[T],
        mode: Optional[ProcessingMode],
        config_overrides: Optional[Dict[str, Any]]
    ) -> ExtractionRequest[T]:
        """Prepare extraction request from inputs"""
        
        # Process content
        if isinstance(content, Path):
            if content.is_file():
                content_text = await self._read_file_content(content)
            elif content.is_dir():
                raise ValueError("Directory content requires extract_from_repository method")
            else:
                raise ValueError(f"Path does not exist: {content}")
        else:
            content_text = content
        
        # Apply configuration overrides
        request_config = {}
        if config_overrides:
            request_config.update(config_overrides)
        
        # Create request
        return ExtractionRequest[T](
            schema_model=schema,
            content=content_text,
            mode=mode or self.config.extraction.default_mode,
            confidence_threshold=self.config.extraction.min_confidence_threshold,
            temperature=request_config.get("temperature", 0.1),
            **{k: v for k, v in request_config.items() if k not in ["temperature"]}
        )
    
    async def _get_or_create_session(self, session_id: Optional[str]) -> ExtractionSession:
        """Get existing session or create new one"""
        if session_id:
            if session_id in self.sessions:
                return self.sessions[session_id]
            else:
                session = ExtractionSession(session_id=session_id)
                self.sessions[session_id] = session
                return session
        else:
            # Use current session or create new one
            if not self.current_session or not self.current_session.active:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.current_session = ExtractionSession(session_id=session_id)
                self.sessions[session_id] = self.current_session
            return self.current_session
    
    async def _check_cache(self, request: ExtractionRequest[T]) -> Optional[ExtractionResult[T]]:
        """Check cache for existing result"""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Compare timestamps consistently
            if cached_data["expires_at"] > datetime.now().timestamp():
                return cached_data["result"]
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
        
        return None
    
    async def _cache_result(self, request: ExtractionRequest[T], result: ExtractionResult[T]):
        """Cache extraction result"""
        cache_key = self._generate_cache_key(request)
        expires_at = datetime.now().timestamp() + self.config.extraction.cache_ttl_seconds
        
        self.cache[cache_key] = {
            "result": result,
            "expires_at": expires_at,  # Store as timestamp (float)
            "created_at": datetime.now().timestamp()  # Store as timestamp for consistency
        }
    
    def _generate_cache_key(self, request: ExtractionRequest[T]) -> str:
        """Generate cache key for request"""
        key_data = {
            "schema": request.schema_model.__name__,
            "content_hash": hashlib.md5(str(request.content).encode()).hexdigest(),
            "mode": request.mode.value,
            "temperature": request.temperature,
            "confidence_threshold": request.confidence_threshold
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def _read_file_content(self, file_path: Path) -> str:
        """Read content from file"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return file_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise DataMinerError(f"Could not decode file: {file_path}")
    
    async def _initialize_llm(self):
        """Initialize LLM integration"""
        try:
            from src.llm import LLMManager
            # Create LLM manager instance directly
            self.llm_manager = LLMManager()
            self.logger.info("LLM integration initialized")
        except ImportError:
            self.logger.warning("LLM module not available")
        except Exception as e:
            raise IntegrationError(f"Failed to initialize LLM: {e}", integration_type="llm")
    
    async def _initialize_mcp(self):
        """Initialize MCP integration"""
        try:
            from src.mcp import FilesystemServer
            self.mcp_server = FilesystemServer()
            await self.mcp_server.initialize()
            self.logger.info("MCP integration initialized")
        except ImportError:
            self.logger.warning("MCP module not available")
        except Exception as e:
            raise IntegrationError(f"Failed to initialize MCP: {e}", integration_type="mcp")
    
    async def _initialize_copilot(self):
        """Initialize Copilot integration"""
        try:
            from src.copilot import CopilotWorkflow
            self.copilot_workflow = CopilotWorkflow()
            await self.copilot_workflow.initialize()
            self.logger.info("Copilot integration initialized")
        except ImportError:
            self.logger.warning("Copilot module not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Copilot: {e}")
            # Don't raise error for Copilot - it's optional
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.log_file:
            # Create log directory if it doesn't exist
            log_dir = Path(self.config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            handler = logging.FileHandler(self.config.log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()